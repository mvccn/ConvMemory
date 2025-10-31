use std::io::BufRead;

use serde_json::Value;
use thiserror::Error;
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;

use crate::types::*;

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("missing field '{0}'")]
    MissingField(&'static str),
    #[error("invalid timestamp '{0}': {1}")]
    Timestamp(String, time::error::Parse),
}

/// Parse a rollout JSONL stream into a structured representation.
pub fn parse_rollout<R: BufRead>(reader: R) -> Result<ConversationRecord, ParseError> {
    let mut builder = ConversationBuilder::default();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line)?;
        if let Some(record_type) = value.get("record_type").and_then(Value::as_str) {
            if record_type == "state" {
                continue;
            }
        }

        let timestamp = if let Some(timestamp_str) = value.get("timestamp").and_then(Value::as_str)
        {
            let parsed = OffsetDateTime::parse(timestamp_str, &Rfc3339)
                .map_err(|err| ParseError::Timestamp(timestamp_str.to_string(), err))?;
            builder.observe_timestamp(parsed);
            parsed
        } else if let Some(last) = builder.last_timestamp {
            last
        } else if let Some(first) = builder.first_timestamp {
            first
        } else {
            return Err(ParseError::MissingField("timestamp"));
        };
        let item_type = match value.get("type").and_then(Value::as_str) {
            Some(kind) => kind,
            None if is_legacy_session_meta(&value) => {
                builder.session_meta = Some(value);
                continue;
            }
            None => return Err(ParseError::MissingField("type")),
        };

        match item_type {
            "session_meta" => {
                builder.session_meta = value
                    .get("payload")
                    .cloned()
                    .or_else(|| Some(value.clone()));
            }
            "turn_context" => {
                if let Some(payload) = value.get("payload") {
                    let context = parse_turn_context(payload.clone());
                    builder.start_new_turn(context, timestamp);
                }
            }
            "response_item" => {
                if let Some(payload) = value.get("payload") {
                    handle_response_item(&mut builder, timestamp, payload.clone());
                }
            }
            "event_msg" => {
                if let Some(payload) = value.get("payload") {
                    handle_event(&mut builder, timestamp, payload.clone());
                }
            }
            "compacted" => {
                if let Some(payload) = value.get("payload") {
                    handle_compacted(&mut builder, timestamp, payload.clone());
                }
            }
            _ => {}
        }
    }
    Ok(builder.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_rollout() {
        let data = r#"
{"timestamp":"2025-01-01T00:00:00.000Z","type":"session_meta","payload":{"id":"urn:uuid:test","cwd":"/tmp"}}
{"timestamp":"2025-01-01T00:00:01.000Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}}
{"timestamp":"2025-01-01T00:00:02.000Z","type":"response_item","payload":{"type":"reasoning","summary":[{"type":"summary_text","text":"thinking"}]}}
{"timestamp":"2025-01-01T00:00:03.000Z","type":"response_item","payload":{"type":"function_call","name":"shell","call_id":"call-1","arguments":"{\"command\":[\"ls\"]}"}}
{"timestamp":"2025-01-01T00:00:04.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"call-1","output":"{\"content\":\"done\"}"}}
{"timestamp":"2025-01-01T00:00:05.000Z","type":"event_msg","payload":{"type":"token_count","rate_limits":{"primary":{"used_percent":1,"window_minutes":1,"resets_at":0}}}}
        "#;

        let cursor = std::io::Cursor::new(data.as_bytes());
        let record = parse_rollout(cursor).expect("parse");
        assert_eq!(record.turns.len(), 1);
        assert_eq!(record.duration_seconds, Some(5));
        assert!(record.token_usage.total.is_none());
        let turn = &record.turns[0];
        assert_eq!(turn.user_inputs.len(), 1);
        assert_eq!(turn.result.reasoning_summaries.len(), 1);
        assert_eq!(turn.actions.len(), 1);
        assert_eq!(turn.actions[0].call_id.as_deref(), Some("call-1"));
        assert_eq!(turn.telemetry.token_counts.len(), 1);
    }
}

fn parse_turn_context(raw: Value) -> TurnContextInfo {
    let cwd = raw
        .get("cwd")
        .or_else(|| raw.get("cwd_path"))
        .and_then(Value::as_str)
        .map(|s| s.to_string());
    let approval_policy = raw
        .get("approval_policy")
        .and_then(Value::as_str)
        .map(|s| s.to_string());
    let sandbox_mode = raw
        .get("sandbox_policy")
        .and_then(|p| p.get("mode"))
        .and_then(Value::as_str)
        .map(|s| s.to_string());
    let sandbox_network_access = raw
        .get("sandbox_policy")
        .and_then(|p| p.get("network_access"))
        .and_then(Value::as_bool);
    let model = raw
        .get("model")
        .and_then(Value::as_str)
        .map(|s| s.to_string());
    let effort = raw
        .get("effort")
        .and_then(Value::as_str)
        .map(|s| s.to_string());
    let summary_style = raw
        .get("summary")
        .and_then(Value::as_str)
        .map(|s| s.to_string());

    TurnContextInfo {
        raw,
        cwd,
        approval_policy,
        sandbox_mode,
        sandbox_network_access,
        model,
        effort,
        summary_style,
    }
}

fn handle_response_item(
    builder: &mut ConversationBuilder,
    timestamp: OffsetDateTime,
    payload: Value,
) {
    let turn = builder.ensure_turn(timestamp);
    turn.ensure_started_at(timestamp);

    let response_type = payload
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    match response_type {
        "message" => handle_message(turn, payload),
        "reasoning" => handle_reasoning(turn, &payload),
        "function_call" => handle_function_call(turn, timestamp, &payload),
        "function_call_output" => handle_function_output(turn, &payload),
        "custom_tool_call" => handle_custom_tool_call(turn, &payload),
        "custom_tool_call_output" => handle_custom_tool_output(turn, &payload),
        "local_shell_call" => handle_local_shell_call(turn, &payload),
        "web_search_call" => handle_web_search_call(turn, &payload),
        _ => {}
    }
}

fn handle_message(turn: &mut TurnBuilder, payload: Value) {
    let role = payload
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let content = payload
        .get("content")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    if role == "user" {
        let mut text_parts = Vec::new();
        let mut images = Vec::new();
        for item in &content {
            match item.get("type").and_then(Value::as_str).unwrap_or_default() {
                "input_text" => {
                    if let Some(text) = item.get("text").and_then(Value::as_str) {
                        text_parts.push(text.to_string());
                    }
                }
                "input_image" => {
                    if let Some(url) = item.get("image_url").and_then(Value::as_str) {
                        images.push(url.to_string());
                    }
                }
                _ => {}
            }
        }
        let record = UserInputRecord {
            raw: payload,
            text: if text_parts.is_empty() {
                None
            } else {
                Some(text_parts.join(""))
            },
            images,
        };
        turn.push_user_input(record);
    } else if role == "assistant" {
        let mut text_parts = Vec::new();
        for item in content {
            if let Some(text) = item.get("text").and_then(Value::as_str) {
                text_parts.push(text.to_string());
            } else if let Some(text) = item.get("content").and_then(Value::as_str) {
                text_parts.push(text.to_string());
            }
        }
        if !text_parts.is_empty() {
            turn.push_assistant_message(text_parts.join(""));
        }
    }
}

fn handle_reasoning(turn: &mut TurnBuilder, payload: &Value) {
    if let Some(summary_items) = payload.get("summary").and_then(Value::as_array) {
        for item in summary_items {
            if let Some(text) = item.get("text").and_then(Value::as_str) {
                turn.push_reasoning_summary(text.to_string());
            }
        }
    }
    if payload.get("content").is_some() {
        turn.mark_reasoning_encrypted();
    }
}

fn handle_function_call(turn: &mut TurnBuilder, timestamp: OffsetDateTime, payload: &Value) {
    let name = payload
        .get("name")
        .and_then(Value::as_str)
        .map(|s| s.to_string());
    let call_id = payload.get("call_id").and_then(Value::as_str);
    let arguments_str = payload
        .get("arguments")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let arguments = serde_json::from_str::<Value>(arguments_str).ok();

    let builder = turn.action_builder_mut(call_id);
    if let Some(name_str) = &name {
        if name_str == "shell" || name_str == "container.exec" {
            let command = arguments
                .as_ref()
                .and_then(|args| args.get("command"))
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(Value::as_str)
                        .map(String::from)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let workdir = arguments
                .as_ref()
                .and_then(|args| args.get("workdir"))
                .or_else(|| {
                    arguments
                        .as_ref()
                        .and_then(|args| args.get("working_directory"))
                })
                .and_then(Value::as_str)
                .map(String::from);
            let timeout_ms = arguments
                .as_ref()
                .and_then(|args| args.get("timeout_ms"))
                .or_else(|| arguments.as_ref().and_then(|args| args.get("timeout")))
                .and_then(Value::as_u64);
            let escalated = arguments
                .as_ref()
                .and_then(|args| args.get("with_escalated_permissions"))
                .and_then(Value::as_bool);

            builder.set_kind(ActionKind::LocalShellExec {
                command,
                workdir,
                timeout_ms,
                escalated,
            });
        } else {
            builder.set_kind(ActionKind::FunctionCall {
                name: Some(name_str.clone()),
            });
        }
    } else {
        builder.set_kind(ActionKind::FunctionCall { name: None });
    }

    builder.set_arguments(arguments);
    builder.push_event(timestamp, "function_call".into(), payload.clone());
}

fn handle_function_output(turn: &mut TurnBuilder, payload: &Value) {
    let call_id = payload.get("call_id").and_then(Value::as_str);
    let output_str = payload
        .get("output")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let raw_output = serde_json::from_str::<Value>(output_str).unwrap_or_else(|_| {
        serde_json::json!({
            "content": output_str,
        })
    });
    let content_text = raw_output
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or(output_str)
        .to_string();

    let builder = turn.action_builder_mut(call_id);
    builder.set_output(ActionOutput {
        content: Some(content_text.clone()),
        success: raw_output.get("success").and_then(Value::as_bool),
        raw: raw_output,
        ..Default::default()
    });
    turn.record_tool_output_text(content_text);
}

fn handle_custom_tool_call(turn: &mut TurnBuilder, payload: &Value) {
    let call_id = payload.get("call_id").and_then(Value::as_str);
    let name = payload
        .get("name")
        .and_then(Value::as_str)
        .map(String::from);
    let status = payload
        .get("status")
        .and_then(Value::as_str)
        .map(String::from);
    let input = payload.get("input").and_then(Value::as_str).unwrap_or("");
    let parsed_input = serde_json::from_str::<Value>(input).ok();

    let builder = turn.action_builder_mut(call_id);
    builder.set_kind(ActionKind::CustomToolCall { name });
    builder.set_arguments(parsed_input);
    builder.update_status_text(status);
}

fn handle_custom_tool_output(turn: &mut TurnBuilder, payload: &Value) {
    let call_id = payload.get("call_id").and_then(Value::as_str);
    let output = payload
        .get("output")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let builder = turn.action_builder_mut(call_id);
    builder.set_output(ActionOutput {
        content: Some(output.clone()),
        success: None,
        raw: Value::String(output.clone()),
    });
    turn.record_tool_output_text(output);
}

fn handle_local_shell_call(turn: &mut TurnBuilder, payload: &Value) {
    let call_id = payload.get("call_id").and_then(Value::as_str);
    let status = payload
        .get("status")
        .and_then(Value::as_str)
        .map(String::from);
    let action = payload.get("action").cloned().unwrap_or(Value::Null);
    let command = action
        .get("command")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(String::from)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let workdir = action
        .get("working_directory")
        .or_else(|| action.get("workdir"))
        .and_then(Value::as_str)
        .map(String::from);
    let timeout_ms = action.get("timeout_ms").and_then(Value::as_u64);
    let escalated = action
        .get("with_escalated_permissions")
        .and_then(Value::as_bool);

    let builder = turn.action_builder_mut(call_id);
    builder.set_kind(ActionKind::LocalShellExec {
        command,
        workdir,
        timeout_ms,
        escalated,
    });
    builder.update_status_text(status.clone());
    builder.update_local_status(status);
    builder.set_arguments(Some(action));
}

fn handle_web_search_call(turn: &mut TurnBuilder, payload: &Value) {
    let call_id = payload.get("call_id").and_then(Value::as_str);
    let status = payload
        .get("status")
        .and_then(Value::as_str)
        .map(String::from);
    let query = payload
        .get("action")
        .and_then(|action| action.get("query"))
        .and_then(Value::as_str)
        .map(String::from);
    let builder = turn.action_builder_mut(call_id);
    builder.set_kind(ActionKind::WebSearch {
        query: query.clone(),
    });
    builder.set_arguments(payload.get("action").cloned());
    builder.update_status_text(status);
}

fn handle_event(builder: &mut ConversationBuilder, timestamp: OffsetDateTime, payload: Value) {
    let event_type = payload
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    let info_for_conversation = if event_type == "token_count" {
        payload.get("info").cloned()
    } else {
        None
    };

    {
        let turn = builder.ensure_turn(timestamp);
        turn.ensure_started_at(timestamp);

        match event_type.as_str() {
            "agent_message" => {
                if let Some(message) = payload.get("message").and_then(Value::as_str) {
                    turn.record_event_agent_message(message.to_string());
                }
                turn.telemetry.misc_events.push(Timed {
                    timestamp,
                    data: payload.clone(),
                });
            }
            "agent_reasoning" | "agent_reasoning_raw_content" => {
                if let Some(text) = payload.get("text").and_then(Value::as_str) {
                    turn.record_event_agent_message(text.to_string());
                }
                turn.telemetry.misc_events.push(Timed {
                    timestamp,
                    data: payload.clone(),
                });
            }
            "token_count" => {
                turn.telemetry.token_counts.push(Timed {
                    timestamp,
                    data: payload.clone(),
                });
            }
            "plan_update" => {
                turn.telemetry.plan_updates.push(Timed {
                    timestamp,
                    data: payload.clone(),
                });
            }
            "exec_approval_request" | "apply_patch_approval_request" => {
                turn.telemetry.approvals.push(Timed {
                    timestamp,
                    data: payload.clone(),
                });
            }
            "exec_command_begin"
            | "exec_command_end"
            | "mcp_tool_call_begin"
            | "mcp_tool_call_end"
            | "web_search_begin"
            | "web_search_end" => {
                let call_id = extract_call_id(&payload);
                let builder = turn.action_builder_mut(call_id.as_deref());
                builder.push_event(timestamp, event_type, payload.clone());
            }
            _ => {
                turn.telemetry.misc_events.push(Timed {
                    timestamp,
                    data: payload.clone(),
                });
            }
        }
    }

    if let Some(info) = info_for_conversation.as_ref() {
        builder.update_token_usage(info);
    }
}

fn handle_compacted(builder: &mut ConversationBuilder, timestamp: OffsetDateTime, payload: Value) {
    let turn = builder.ensure_turn(timestamp);
    turn.ensure_started_at(timestamp);
    if let Some(message) = payload.get("message").and_then(Value::as_str) {
        turn.push_assistant_message(message.to_string());
        turn.record_tool_output_text(message.to_string());
    }
}

fn extract_call_id(payload: &Value) -> Option<String> {
    payload
        .get("call_id")
        .or_else(|| payload.get("callId"))
        .and_then(Value::as_str)
        .map(String::from)
}

fn is_legacy_session_meta(value: &Value) -> bool {
    value.get("type").is_none()
        && value.get("record_type").is_none()
        && value.get("id").is_some()
        && value.get("timestamp").is_some()
}
