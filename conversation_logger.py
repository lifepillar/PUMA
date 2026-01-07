import sqlite3

from chat_round import ChatRound
from datetime import datetime
from pathlib import Path
from protocols import LanguageModelResponse


class ConversationLogger:  # Conforms to SolutionLogger and LanguageModelLogger
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.con = sqlite3.connect(path)
        self._create_schema()

    def response(self, response: LanguageModelResponse) -> None:
        """Log a response."""
        current_datetime = datetime.now()
        created_at = int(current_datetime.timestamp())  # Unix timestamp

        with self.con as con:
            con.execute(
                "insert or ignore into Conversation"
                "(Conversation, SystemPrompt, Model, Role, DateTime, CreatedAt) "
                "values (:conversation, :system_prompt, :model, :role, :datetime, :created_at)",
                {
                    "conversation": response.conversation_id,
                    "system_prompt": response.system_prompt(),
                    "model": response.model,
                    "role": response.role,
                    "datetime": str(current_datetime),
                    "created_at": created_at,
                },
            )
            con.execute(
                "insert into Response"
                "(Conversation, Round, Prompt, Response, Reasoning, DateTime, CreatedAt) "
                "values "
                "(:conversation, :round, :prompt, :response, :reasoning, :datetime, :created_at)",
                {
                    "conversation": response.conversation_id,
                    "round": response.round,
                    "prompt": response.prompt(),
                    "response": response.text(),
                    "reasoning": response.reasoning(),
                    "datetime": str(current_datetime),
                    "created_at": created_at,
                },
            )

    def solution(self, solution: ChatRound) -> None:
        """Log a solution.

        Log the generation number of the solution and the relationship between
        a response and its parent responses.
        """
        with self.con as con:
            for parent in solution.parents():
                con.execute(
                    "insert or ignore into ParentOffspring"
                    "(Conversation, Round, Generation, ParentConversation, ParentRound) "
                    "values "
                    "(:conversation, :round, :generation, :parent, :parent_round)",
                    {
                        "conversation": solution.conversation_id,
                        "round": solution.round,
                        "generation": solution.generation,
                        "parent": parent.conversation_id,
                        "parent_round": parent.round,
                    },
                )

    def evaluation(self, solution: ChatRound) -> None:
        """Log an evaluation result."""
        current_datetime = datetime.now()
        created_at = int(current_datetime.timestamp())  # Unix timestamp

        with self.con as con:
            con.execute(
                "insert or ignore into Evaluation"
                "(Conversation, Round, Fitness, Feedback, DateTime, CreatedAt) "
                "values "
                "(:conversation, :round, :fitness, :feedback, :datetime, :created_at)",
                {
                    "conversation": solution.conversation_id,
                    "round": solution.round,
                    "fitness": solution.fitness,
                    "feedback": solution.feedback,
                    "datetime": str(current_datetime),
                    "created_at": created_at,
                },
            )

    def feature(self, solution: ChatRound, feature_name: str, value: float) -> None:
        with self.con as con:
            con.execute(
                "insert or ignore into Feature"
                "(Conversation, Round, Name, Value) "
                "values "
                "(:conversation, :round, :name, :value)",
                {
                    "conversation": solution.conversation_id,
                    "round": solution.round,
                    "name": feature_name,
                    "value": value,
                },
            )

    def dump(self) -> list[dict[str, float | int | str]]:
        """Dump the log."""
        attributes = [
            "DateTime",
            "Model",
            "Role",
            "Conversation",
            "Round",
            "Prompt",
            "Response",
            "Reasoning",
            "Fitness",
            "Feedback",
        ]
        with self.con as con:
            result_set: list[tuple] = con.execute(
                f"select {','.join(attributes)} "
                "from Log "
                "order by DateTime desc, Conversation, Round"
            ).fetchall()

        return [{attributes[i]: v for i, v in enumerate(t)} for t in result_set]

    def _create_schema(self) -> None:
        with self.con as con:
            con.executescript("""
                begin;
                create table if not exists Conversation (
                  Conversation       text not null,
                  SystemPrompt       text not null default '',
                  Model              text not null,
                  Role               text not null default 'default',
                  DateTime           text not null,
                  CreatedAt          int  not null, -- Same as DateTime, Unix timestamp

                  primary key (Conversation)
                );

                create table if not exists Response (
                  Conversation       text not null,
                  Round              int  not null,
                  Prompt             text not null,
                  Response           text not null,
                  Reasoning          text not null  default '',
                  DateTime           text not null,
                  CreatedAt          int  not null, -- Same as DateTime, Unix timestamp

                  primary key (Conversation, Round),
                  constraint Conversation_spawns_Responses
                    foreign key (Conversation) references Conversation
                );

                create table if not exists ParentOffspring (
                  Conversation       text not null,
                  Round              int  not null,
                  Generation         int  not null,
                  ParentConversation text not null,
                  ParentRound        int  not null,

                  primary key (Conversation, Round),
                  constraint Response_generates_Offspring
                    foreign key (Conversation, ParentRound)
                      references Response(Conversation, Round),
                  constraint Response_is_generated_by_Parent
                    foreign key (Conversation, Round)
                      references Response(Conversation, Round)
                );

                create table if not exists Evaluation (
                  Conversation       text not null,
                  Round              int  not null,
                  Fitness            real not null,
                  Feedback           text not null default '',
                  DateTime           text not null,
                  CreatedAt          int  not null, -- Same as DateTime, Unix timestamp

                  primary key (Conversation, Round),
                  constraint Response_gets_Evaluation
                    foreign key (Conversation, Round) references Response
                );

                create table if not exists Feature (
                  Conversation       text not null,
                  Round              int  not null,
                  Name               text not null,
                  Value              real not null,

                  primary key (Conversation, Round, Name),
                  constraint Response_has_Feature
                    foreign key (Conversation, Round) references Response
                );

                create view if not exists Summary as
                select C.DateTime as ConversationTime, C.Model, C.Role, C.Conversation,
                       R.Round, R.Prompt, R.Response, R.Reasoning,
                       coalesce(E.fitness, '—') as Fitness, coalesce(E.feedback, '—') as Feedback,
                       R.DateTime as ResponseTime, R.CreatedAt
                from Conversation C join Response R on C.Conversation = R.Conversation
                left join Evaluation E on E.Conversation = C.Conversation and E.Round = R.Round;

                commit;
                """)
