use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use time::OffsetDateTime;

/// Parsed representation of a rollout file.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConversationRecord {
    pub session_meta: Option<Value>,
    pub started_at: Option<OffsetDateTime>,
    pub ended_at: Option<OffsetDateTime>,
    pub duration_seconds: Option<u64>,
    pub token_usage: TokenUsageSummary,
    pub turns: Vec<TurnRecord>,
}

/// Normalised view of a single turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnRecord {
    pub index: usize,
    pub started_at: Option<OffsetDateTime>,
    pub context: Option<TurnContextInfo>,
    pub user_inputs: Vec<UserInputRecord>,
    pub result: TurnResult,
    pub actions: Vec<ActionRecord>,
    pub telemetry: TurnTelemetry,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnContextInfo {
    pub raw: Value,
    pub cwd: Option<String>,
    pub approval_policy: Option<String>,
    pub sandbox_mode: Option<String>,
    pub sandbox_network_access: Option<bool>,
    pub model: Option<String>,
    pub effort: Option<String>,
    pub summary_style: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInputRecord {
    pub raw: Value,
    pub text: Option<String>,
    pub images: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TurnResult {
    pub assistant_messages: Vec<String>,
    pub fallback: Option<FallbackSummary>,
    pub reasoning_summaries: Vec<String>,
    pub reasoning_encrypted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackSummary {
    pub source: FallbackSource,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FallbackSource {
    AssistantReasoning,
    ToolOutput,
    EventStream,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ActionRecord {
    pub call_id: Option<String>,
    pub kind: ActionKind,
    pub arguments: Option<Value>,
    pub output: Option<ActionOutput>,
    pub status: ActionStatus,
    pub events: Vec<ActionEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionKind {
    FunctionCall {
        name: Option<String>,
    },
    CustomToolCall {
        name: Option<String>,
    },
    LocalShellExec {
        command: Vec<String>,
        workdir: Option<String>,
        timeout_ms: Option<u64>,
        escalated: Option<bool>,
    },
    WebSearch {
        query: Option<String>,
    },
    Other {
        kind: Option<String>,
    },
}

impl Default for ActionKind {
    fn default() -> Self {
        ActionKind::Other { kind: None }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ActionOutput {
    pub content: Option<String>,
    pub success: Option<bool>,
    pub raw: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ActionStatus {
    pub status_text: Option<String>,
    pub local_status: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionEvent {
    pub timestamp: OffsetDateTime,
    pub kind: String,
    pub data: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TurnTelemetry {
    pub token_counts: Vec<Timed<Value>>,
    pub plan_updates: Vec<Timed<Value>>,
    pub approvals: Vec<Timed<Value>>,
    pub misc_events: Vec<Timed<Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timed<T> {
    pub timestamp: OffsetDateTime,
    pub data: T,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenUsageSummary {
    pub total: Option<TokenUsageBreakdown>,
    pub last: Option<TokenUsageBreakdown>,
    pub model_context_window: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenUsageBreakdown {
    pub input_tokens: Option<u64>,
    pub cached_input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub reasoning_output_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
}

/// Helper used while constructing a conversation record.
#[derive(Default)]
pub(crate) struct ConversationBuilder {
    pub session_meta: Option<Value>,
    pub turns: Vec<TurnRecord>,
    pub current_turn: Option<TurnBuilder>,
    pub next_index: usize,
    pub first_timestamp: Option<OffsetDateTime>,
    pub last_timestamp: Option<OffsetDateTime>,
    pub token_usage: TokenUsageSummary,
}

#[derive(Default)]
pub(crate) struct TurnBuilder {
    pub index: usize,
    pub started_at: Option<OffsetDateTime>,
    pub context: Option<TurnContextInfo>,
    pub user_inputs: Vec<UserInputRecord>,
    pub assistant_messages: Vec<String>,
    pub reasoning_summaries: Vec<String>,
    pub reasoning_encrypted: bool,
    pub fallback_reasoning: Option<String>,
    pub fallback_tool_output: Option<String>,
    pub fallback_event: Option<String>,
    pub actions: HashMap<String, ActionRecordBuilder>,
    pub anonymous_actions: Vec<ActionRecordBuilder>,
    pub telemetry: TurnTelemetry,
}

impl ConversationBuilder {
    pub fn observe_timestamp(&mut self, timestamp: OffsetDateTime) {
        if self.first_timestamp.is_none() {
            self.first_timestamp = Some(timestamp);
        }
        self.last_timestamp = Some(timestamp);
    }

    pub fn update_token_usage(&mut self, info: &Value) {
        if let Some(total) = info.get("total_token_usage") {
            self.token_usage.total = Some(TokenUsageBreakdown::from_value(total));
        }
        if let Some(last) = info.get("last_token_usage") {
            self.token_usage.last = Some(TokenUsageBreakdown::from_value(last));
        }
        if let Some(window) = info
            .get("model_context_window")
            .or_else(|| info.get("model_context_window_tokens"))
            .and_then(Value::as_u64)
        {
            self.token_usage.model_context_window = Some(window);
        }
    }

    pub fn ensure_turn(&mut self, timestamp: OffsetDateTime) -> &mut TurnBuilder {
        if self.current_turn.is_none() {
            let index = self.next_index;
            self.next_index += 1;
            self.current_turn = Some(TurnBuilder {
                index,
                started_at: Some(timestamp),
                ..TurnBuilder::default()
            });
        }
        self.current_turn.as_mut().unwrap()
    }

    pub fn start_new_turn(
        &mut self,
        context: TurnContextInfo,
        timestamp: OffsetDateTime,
    ) -> &mut TurnBuilder {
        if let Some(builder) = self.current_turn.take() {
            if !builder.is_empty() {
                self.turns.push(builder.finish());
            }
        }
        let index = self.next_index;
        self.next_index += 1;
        self.current_turn = Some(TurnBuilder {
            index,
            started_at: Some(timestamp),
            context: Some(context),
            ..TurnBuilder::default()
        });
        self.current_turn.as_mut().unwrap()
    }

    pub fn finalize(mut self) -> ConversationRecord {
        if let Some(builder) = self.current_turn.take() {
            if !builder.is_empty() {
                self.turns.push(builder.finish());
            }
        }
        let duration_seconds = match (self.first_timestamp, self.last_timestamp) {
            (Some(start), Some(end)) => Some((end - start).whole_seconds().max(0) as u64),
            _ => None,
        };

        ConversationRecord {
            session_meta: self.session_meta,
            started_at: self.first_timestamp,
            ended_at: self.last_timestamp,
            duration_seconds,
            token_usage: self.token_usage,
            turns: self.turns,
        }
    }
}

impl TurnBuilder {
    pub fn ensure_started_at(&mut self, timestamp: OffsetDateTime) {
        if self.started_at.is_none() {
            self.started_at = Some(timestamp);
        }
    }

    pub fn push_user_input(&mut self, input: UserInputRecord) {
        self.user_inputs.push(input);
    }

    pub fn push_assistant_message(&mut self, message: String) {
        self.assistant_messages.push(message);
    }

    pub fn push_reasoning_summary(&mut self, summary: String) {
        self.reasoning_summaries.push(summary.clone());
        self.fallback_reasoning = Some(summary);
    }

    pub fn mark_reasoning_encrypted(&mut self) {
        self.reasoning_encrypted = true;
    }

    pub fn record_tool_output_text(&mut self, text: String) {
        self.fallback_tool_output = Some(text);
    }

    pub fn record_event_agent_message(&mut self, text: String) {
        self.fallback_event = Some(text);
    }

    pub fn action_builder_mut(&mut self, call_id: Option<&str>) -> &mut ActionRecordBuilder {
        if let Some(id) = call_id {
            self.actions
                .entry(id.to_string())
                .or_insert_with(|| ActionRecordBuilder::new(Some(id.to_string())))
        } else {
            self.anonymous_actions.push(ActionRecordBuilder::new(None));
            self.anonymous_actions
                .last_mut()
                .expect("anonymous action builder present")
        }
    }

    pub fn is_empty(&self) -> bool {
        self.user_inputs.is_empty()
            && self.assistant_messages.is_empty()
            && self.actions.is_empty()
            && self.anonymous_actions.is_empty()
            && self.reasoning_summaries.is_empty()
            && self.telemetry.token_counts.is_empty()
    }

    pub fn finish(mut self) -> TurnRecord {
        let mut actions: Vec<ActionRecord> = self
            .actions
            .into_iter()
            .map(|(_, builder)| builder.finish())
            .collect();
        actions.extend(self.anonymous_actions.into_iter().map(|b| b.finish()));
        actions.sort_by(|a, b| a.call_id.cmp(&b.call_id));

        let fallback = if !self.assistant_messages.is_empty() {
            None
        } else if let Some(text) = self.fallback_reasoning.take() {
            Some(FallbackSummary {
                source: FallbackSource::AssistantReasoning,
                text,
            })
        } else if let Some(text) = self.fallback_tool_output.take() {
            Some(FallbackSummary {
                source: FallbackSource::ToolOutput,
                text,
            })
        } else if let Some(text) = self.fallback_event.take() {
            Some(FallbackSummary {
                source: FallbackSource::EventStream,
                text,
            })
        } else {
            None
        };

        TurnRecord {
            index: self.index,
            started_at: self.started_at,
            context: self.context,
            user_inputs: self.user_inputs,
            result: TurnResult {
                assistant_messages: self.assistant_messages,
                fallback,
                reasoning_summaries: self.reasoning_summaries,
                reasoning_encrypted: self.reasoning_encrypted,
            },
            actions,
            telemetry: self.telemetry,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ActionRecordBuilder {
    call_id: Option<String>,
    kind: ActionKind,
    arguments: Option<Value>,
    output: Option<ActionOutput>,
    status: ActionStatus,
    events: Vec<ActionEvent>,
}

impl ActionRecordBuilder {
    pub fn new(call_id: Option<String>) -> Self {
        Self {
            call_id,
            ..Self::default()
        }
    }

    pub fn set_kind(&mut self, kind: ActionKind) {
        self.kind = kind;
    }

    pub fn set_arguments(&mut self, args: Option<Value>) {
        self.arguments = args;
    }

    pub fn set_output(&mut self, output: ActionOutput) {
        self.output = Some(output);
    }

    pub fn update_status_text(&mut self, status: Option<String>) {
        self.status.status_text = status;
    }

    pub fn update_local_status(&mut self, status: Option<String>) {
        self.status.local_status = status;
    }

    pub fn push_event(&mut self, timestamp: OffsetDateTime, kind: String, data: Value) {
        self.events.push(ActionEvent {
            timestamp,
            kind,
            data,
        });
    }

    pub fn finish(self) -> ActionRecord {
        ActionRecord {
            call_id: self.call_id,
            kind: self.kind,
            arguments: self.arguments,
            output: self.output,
            status: self.status,
            events: self.events,
        }
    }
}

impl TokenUsageBreakdown {
    pub fn from_value(value: &Value) -> Self {
        TokenUsageBreakdown {
            input_tokens: value.get("input_tokens").and_then(Value::as_u64),
            cached_input_tokens: value
                .get("cached_input_tokens")
                .or_else(|| value.get("cachedTokens"))
                .and_then(Value::as_u64),
            output_tokens: value.get("output_tokens").and_then(Value::as_u64),
            reasoning_output_tokens: value.get("reasoning_output_tokens").and_then(Value::as_u64),
            total_tokens: value.get("total_tokens").and_then(Value::as_u64),
        }
    }
}
