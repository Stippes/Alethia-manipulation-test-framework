import base64
import io
import json
from datetime import datetime
from typing import Dict, Any, List
import logging
from logging_utils import setup_logging

from insight_helpers import (
    compute_manipulation_ratio,
    compute_manipulation_timeline,
    compute_most_manipulative_message,
    compute_dominance_metrics,
    compute_llm_flag_timeline,
)

try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output, State
    import plotly.graph_objs as go
    import dash_bootstrap_components as dbc
except Exception:  # pragma: no cover - make optional for tests
    class _Dummy:
        def __getattr__(self, name):
            return self

        def __call__(self, *args, **kwargs):
            return self

    dash = _Dummy()
    dcc = html = Input = Output = State = _Dummy()
    go = _Dummy()
    dbc = _Dummy()

setup_logging()
logger = logging.getLogger(__name__)


# pick one of the Bootswatch themes below:
# ['CERULEAN','COSMO','CYBORG','DARKLY','FLATLY','JOURNAL',
#  'LUMEN','PULSE','SLATE','SOLAR','SPACELAB',
#  'SUPERHERO','UNITED','VAPOR','YETI']

from scripts import input_parser, static_feature_extractor
from scripts.judge_conversation import judge_conversation_llm
from scripts.judge_utils import merge_judge_results
import scorer

# Flags added beyond the original four categories
NEW_FLAGS = [
    ("guilt", "Guilt Trips"),
    ("social_proof", "Social Proof"),
    ("authority", "Authority"),
    ("reciprocity", "Reciprocity"),
    ("consistency", "Consistency"),
    ("dependency", "Dependency"),
    ("fear", "Fear/Threats"),
    ("gaslighting", "Gaslighting"),
    ("deception", "Deception"),
]

ALL_FLAG_NAMES = [
    "urgency",
    "guilt",
    "flattery",
    "fomo",
    "social_proof",
    "authority",
    "reciprocity",
    "consistency",
    "dependency",
    "fear",
    "gaslighting",
    "deception",
    "dark_ui",
    "emotion_count",
]


def compute_flag_counts(features: List[Dict[str, Any]], judge_results: Dict[str, Any]) -> (
    Dict[str, int], Dict[str, int]
):
    """Return heuristic and LLM counts for each flag.

    ``judge_results`` is expected to contain a top-level ``"flagged"`` list,
    typically produced by :func:`merge_judge_results`.
    """
    heur = {f: 0 for f in ALL_FLAG_NAMES}
    for feat in features:
        flags = feat.get("flags", {})
        for f in ALL_FLAG_NAMES:
            if f == "emotion_count":
                heur[f] += int(flags.get(f, 0) or 0)
            elif flags.get(f):
                heur[f] += 1

    llm = {f: 0 for f in ALL_FLAG_NAMES}
    flagged = judge_results.get("flagged") if isinstance(judge_results, dict) else []
    for item in flagged or []:
        flags = item.get("flags", {})
        for f in ALL_FLAG_NAMES:
            if flags.get(f):
                llm[f] += 1
    return heur, llm


def build_flag_comparison_figure(
    features: List[Dict[str, Any]],
    judge_results: Dict[str, Any],
    bg: str,
    text_color: str,
) -> "go.Figure":
    """Create bar chart comparing heuristic vs LLM flag counts."""
    heur, llm = compute_flag_counts(features, judge_results)
    labels = [f.replace("_", " ").title() for f in ALL_FLAG_NAMES]
    return go.Figure(
        data=[
            go.Bar(
                name="Heuristic",
                x=labels,
                y=[heur[f] for f in ALL_FLAG_NAMES],
                marker_color="#17BECF",
            ),
            go.Bar(
                name="LLM",
                x=labels,
                y=[llm[f] for f in ALL_FLAG_NAMES],
                marker_color="#EF553B",
            ),
        ],
        layout=go.Layout(
            title="\U0001F4CA Flag Counts: Heuristic vs LLM",
            barmode="group",
            paper_bgcolor=bg,
            plot_bgcolor=bg,
            font=dict(color=text_color),
            xaxis=dict(title="Flag", color=text_color, gridcolor="#444"),
            yaxis=dict(title="Count", color=text_color, gridcolor="#444"),
        ),
    )


def parse_uploaded_file(contents: str, filename: str) -> Dict[str, Any]:
    logger.debug("Parsing uploaded file %s", filename)
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    name = filename.lower()
    if name.endswith('.json'):
        raw = json.load(io.StringIO(decoded.decode('utf-8')))
        if isinstance(raw, dict) and 'messages' in raw:
            conversation = raw
            conversation.setdefault('conversation_id', filename.rsplit('.', 1)[0])
        else:
            conversation = {'conversation_id': filename.rsplit('.', 1)[0], 'messages': raw if isinstance(raw, list) else []}
    elif name.endswith('.txt'):
        data = decoded.decode('utf-8')
        lines = data.splitlines()
        msgs = []
        for line in lines:
            if not line.strip():
                continue
            if line.startswith('['):
                try:
                    timestamp_part, rest = line.split(']', 1)
                    timestamp = timestamp_part[1:]
                    sender, text = rest.split(':', 1)
                    msgs.append({'sender': sender.strip(), 'timestamp': timestamp.strip(), 'text': text.strip()})
                    continue
                except Exception:
                    pass
            msgs.append({'sender': None, 'timestamp': None, 'text': line})
        conversation = {'conversation_id': filename.rsplit('.', 1)[0], 'messages': msgs}
    elif name.endswith('.csv'):
        data = decoded.decode('utf-8')
        lines = data.splitlines()
        headers = [h.strip().lower() for h in lines[0].split(',')]
        msgs: List[Dict[str, Any]] = []
        for row in lines[1:]:
            cols = row.split(',')
            record = {h: cols[i].strip() if i < len(cols) else '' for i, h in enumerate(headers)}
            msgs.append({'sender': record.get('sender'), 'timestamp': record.get('timestamp'), 'text': record.get('text') or record.get('message', '')})
        conversation = {'conversation_id': filename.rsplit('.', 1)[0], 'messages': msgs}
    else:
        conversation = {'conversation_id': filename.rsplit('.', 1)[0], 'messages': []}
    return input_parser.standardize_format(conversation)


def analyze_conversation(conv: Dict[str, Any]) -> Dict[str, Any]:
    logger.info("Analyzing conversation %s", conv.get('conversation_id'))
    features = static_feature_extractor.extract_conversation_features(conv)
    trust_score = scorer.score_trust(features)
    risk = scorer.compute_risk_score(features)
    summary = {
        'dark_patterns': sum(1 for f in features if f['flags'].get('dark_ui')),
        'emotional_framing': sum(f['flags'].get('emotion_count', 0) for f in features),
        'parasocial_pressure': sum(1 for f in features if f['flags'].get('flattery')),
        'reinforcement_loops': sum(1 for f in features if f['flags'].get('urgency') or f['flags'].get('fomo')),
    }
    for flag, _ in NEW_FLAGS:
        summary[flag] = sum(1 for f in features if f['flags'].get(flag))
    manipulation_ratio = compute_manipulation_ratio(features)
    manipulation_timeline = compute_manipulation_timeline(features)
    most_manipulative = compute_most_manipulative_message(features)
    dominance_metrics = compute_dominance_metrics(features)

    logger.info("Analysis produced risk %s", risk)
    return {
        'features': features,
        'risk': risk,
        'summary': summary,
        'manipulation_ratio': manipulation_ratio,
        'manipulation_timeline': manipulation_timeline,
        'most_manipulative': most_manipulative,
        'dominance_metrics': dominance_metrics,
    }


def summarize_judge_results(judge_results: Dict[str, Any]) -> str:
    if not isinstance(judge_results, dict):
        return ""
    flagged = judge_results.get("flagged", [])
    counts = {f: 0 for f in ALL_FLAG_NAMES}
    for item in flagged:
        for f in ALL_FLAG_NAMES:
            if item.get("flags", {}).get(f):
                counts[f] += 1
    parts = [f"Total flagged: {len(flagged)}"]
    parts.extend(
        f"{f.replace('_', ' ').title()}: {counts[f]}" for f in ALL_FLAG_NAMES if counts[f]
    )
    return "; ".join(parts)


DARK_THEME = dbc.themes.DARKLY
LIGHT_THEME = dbc.themes.FLATLY

external_stylesheets = [dbc.icons.FONT_AWESOME, DARK_THEME]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Alethia Manipulation Transparency Console"

default_figure = go.Figure(
    data=[go.Bar(x=[], y=[], marker_color="#17BECF")],
    layout=go.Layout(
        title="\U0001F4CA Pattern Breakdown",
        paper_bgcolor="#1a1a1a",
        plot_bgcolor="#1a1a1a",
        font=dict(color="white"),
        xaxis=dict(title="Pattern Type", color="white"),
        yaxis=dict(title="Count", color="white"),
    ),
)

default_timeline = go.Figure(
    data=[go.Scatter(x=[], y=[])],
    layout=go.Layout(
        title="\U0001F4CA Manipulation Intensity Over Time",
        paper_bgcolor="#1a1a1a",
        plot_bgcolor="#1a1a1a",
        font=dict(color="white"),
        xaxis=dict(title="Message Index", color="white"),
        yaxis=dict(title="Active Flags", color="white"),
    ),
)

default_comparison = go.Figure(
    data=[go.Bar(x=[], y=[])],
    layout=go.Layout(
        title="\U0001F4CA Flag Counts: Heuristic vs LLM",
        paper_bgcolor="#1a1a1a",
        plot_bgcolor="#1a1a1a",
        font=dict(color="white"),
        xaxis=dict(title="Flag", color="white"),
        yaxis=dict(title="Count", color="white"),
    ),
)


def create_empty_figure(title: str, bg: str, text_color: str) -> "go.Figure":
    return go.Figure(
        layout=go.Layout(
            title=title,
            paper_bgcolor=bg,
            plot_bgcolor=bg,
            font=dict(color=text_color),
        )
    )

app.layout = html.Div([
    html.Link(id="theme-link", rel="stylesheet", href=DARK_THEME),
    dcc.Store(id="theme-store", data="dark"),
    dcc.Store(id="llm-debug", data=[]),
    dcc.Store(id="judge-store"),
    dbc.Container(
        fluid=True,
        children=[
        dbc.Row(
            [
                dbc.Col(
                    html.H1(
                        "Alethia Manipulation Transparency Console",
                        className="text-center text-light my-4",
                        id="top",
                    ),
                    width="auto",
                ),
                dbc.Col(
                    dbc.Switch(
                        id="theme-toggle",
                        label="Light mode",
                        value=False,
                        className="ms-2 mt-4",
                    ),
                    width="auto",
                    align="center",
                ),
            ],
            justify="center",
            className="mb-4",
        ),
        dbc.Row(
            [
                # Sidebar
                dbc.Col(
                    [
                    dbc.Card(
                        [
                            dbc.CardHeader("Controls"),
                            dbc.CardBody(
                                [
                                    dcc.Upload(
                                        id="upload-data",
                                        children=dbc.Button(
                                            "Upload Conversation",
                                            color="primary",
                                            className="mb-3",
                                        ),
                                        multiple=False,
                                    ),
                                    dcc.Dropdown(
                                        id="conv-type",
                                        options=[
                                            {"label": "Chatbot", "value": "chatbot"},
                                            {"label": "Social Media", "value": "social"},
                                        ],
                                        value="chatbot",
                                        className="mb-3",
                                        style={
                                            "backgroundColor": "#2b2b2b",
                                            "color": "#dddddd",
                                            "border": "1px solid #444",
                                        },
                                    ),
                                    dcc.Checklist(
                                        id="pattern-filter",
                                        options=[
                                            {
                                                "label": "Dark Patterns",
                                                "value": "dark_patterns",
                                            },
                                            {
                                                "label": "Emotional Framing",
                                                "value": "emotional_framing",
                                            },
                                            {
                                                "label": "Parasocial Pressure",
                                                "value": "parasocial_pressure",
                                            },
                                            {
                                                "label": "Reinforcement Loops",
                                                "value": "reinforcement_loops",
                                            },
                                            *[
                                                {
                                                    "label": label,
                                                    "value": flag,
                                                }
                                                for flag, label in NEW_FLAGS
                                            ],
                                        ],
                                        value=[
                                            "dark_patterns",
                                            "emotional_framing",
                                            "parasocial_pressure",
                                            "reinforcement_loops",
                                            *[flag for flag, _ in NEW_FLAGS],
                                        ],
                                        inline=False,
                                        className="mb-3 text-light",
                                    ),
                                    dcc.Dropdown(
                                        id="llm-provider",
                                        options=[
                                            {"label": "Auto", "value": "auto"},
                                            {"label": "OpenAI", "value": "openai"},
                                            {"label": "Claude", "value": "claude"},
                                            {"label": "Mistral", "value": "mistral"},
                                            {"label": "Gemini", "value": "gemini"},
                                        ],
                                        value="auto",
                                        className="mb-2",
                                        style={
                                            "backgroundColor": "#2b2b2b",
                                            "color": "#dddddd",
                                            "border": "1px solid #444",
                                        },
                                    ),
                                    dbc.Button(
                                        "Analyze with LLM judge",
                                        id="llm-judge-btn",
                                        color="info",
                                        className="mb-3",
                                    ),
                                    html.Div(id="file-info", className="text-muted mb-2"),
                                    html.H5("\u26A0\ufe0f Manipulation Risk", className="text-light"),
                                    html.Div(
                                        id="risk-score",
                                        className="h3 text-warning mb-3",
                                    ),
                                    dbc.ListGroup(
                                        [
                                            dbc.ListGroupItem(id="dark-patterns", color="dark"),
                                            dbc.ListGroupItem(id="emotional-framing", color="dark"),
                                            dbc.ListGroupItem(id="parasocial-pressure", color="dark"),
                                            dbc.ListGroupItem(id="reinforcement-loops", color="dark"),
                                            *[
                                                dbc.ListGroupItem(id=flag.replace('_', '-'), color="dark")
                                                for flag, _ in NEW_FLAGS
                                            ],
                                        ],
                                        flush=True,
                                    ),
                                ]
                            ),
                        ],
                        className="mb-4 h-100 shadow-sm",
                    ),
                    dbc.Card(
                        [
                            dbc.CardHeader("LLM Flag Summary"),
                            dbc.CardBody(
                                html.Div(id="llm-summary", className="text-light")
                            ),
                        ],
                        className="mb-4 shadow-sm",
                    ),
                    ],
                    width=3,
                ),
                # Conversation & Graphs
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                dbc.RadioItems(
                                    id="view-mode",
                                    options=[
                                        {"label": "Clean", "value": "clean"},
                                        {"label": "Annotated", "value": "annotated"},
                                    ],
                                    value="clean",
                                    inline=True,
                                    labelStyle={"color": "white"},
                                )
                            ),
                            dbc.CardBody(
                                [
                                    html.Div(
                                        id="conversation-view",
                                        style={
                                            "height": "300px",
                                            "overflowY": "auto",
                                            "backgroundColor": "#212529",
                                            "padding": "1rem",
                                            "borderRadius": "0.25rem",
                                        },
                                    ),
                                    dcc.Graph(id="pattern-graph", className="mt-4"),
                                    dcc.Graph(id="manipulation-graph", className="mt-4"),
                                    dcc.Graph(id="flag-comparison", className="mt-4"),
                                    html.Div(id="most-manipulative", className="mt-3 text-light"),
                                    html.Div(id="llm-judge-results", className="mt-3"),
                                    dbc.Button(
                                        "Download JSON Report",
                                        id="download-json-btn",
                                        color="secondary",
                                        className="mt-3",
                                    ),
                                    dcc.Download(id="download-json"),
                                    html.Pre(id="debug-output", className="mt-3 text-light", style={"whiteSpace": "pre-wrap"}),
                                ]
                            ),
                        ],
                        className="mb-4 h-100 shadow-sm",
                    ),
                    width=6,
                ),
                # Explanations & Metrics
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("\U0001F9E0 Explanations & Examples"),
                            dbc.CardBody(
                                [
                                    html.Div(id="explanations"),
                                    html.Hr(className="bg-light"),
                                    html.Div(id="dominance-table"),
                                    html.H5("What You Can Do", className="text-light mt-3"),
                                    dbc.ListGroup(
                                        [
                                            dbc.ListGroupItem(
                                                "Turn off autoplay / limit notifications",
                                                color="dark",
                                            ),
                                            dbc.ListGroupItem(
                                                "Save conversation logs", color="dark"
                                            ),
                                            dbc.ListGroupItem(
                                                "Disable tracking settings", color="dark"
                                            ),
                                        ],
                                        flush=True,
                                    ),
                                ]
                            ),
                        ],
                        className="mb-4 h-100 shadow-sm",
                    ),
                    width=3,
                ),
            ],
            className="g-4 mb-4",
        ),
    ],
        className="mb-4",
    ),
    html.A(
        html.I(className="fa fa-arrow-up"),
        href="#top",
        id="scroll-top",
        className="btn btn-secondary position-fixed bottom-0 end-0 m-3",
    ),
])


@app.callback(
    [
        Output("file-info", "children"),
        Output("risk-score", "children"),
        Output("dark-patterns", "children"),
        Output("emotional-framing", "children"),
        Output("parasocial-pressure", "children"),
        Output("reinforcement-loops", "children"),
        *[
            Output(flag.replace('_', '-'), "children")
            for flag, _ in NEW_FLAGS
        ],
        Output("conversation-view", "children"),
        Output("pattern-graph", "figure"),
        Output("manipulation-graph", "figure"),
        Output("flag-comparison", "figure"),
        Output("most-manipulative", "children"),
        Output("llm-summary", "children"),
        Output("llm-judge-results", "children"),
        Output("dominance-table", "children"),
        Output("explanations", "children"),
        Output("download-json", "data"),
        Output("llm-debug", "data"),
        Output("judge-store", "data"),
    ],
    [
        Input("upload-data", "contents"),
        Input("view-mode", "value"),
        Input("download-json-btn", "n_clicks"),
        Input("llm-judge-btn", "n_clicks"),
        Input("llm-provider", "value"),
        Input("pattern-filter", "value"),
    ],
    [
        State("upload-data", "filename"),
        State("theme-toggle", "value"),
        State("llm-debug", "data"),
        State("judge-store", "data"),
    ],
)

def update_output(
    contents,
    view_mode,
    download_clicks,
    judge_clicks,
    provider,
    selected_patterns,
    filename,
    light_on,
    debug_log,
    judge_data,
):
    bg = "#ffffff" if light_on else "#1a1a1a"
    text_color = "black" if light_on else "white"
    log_entries = list(debug_log or [])
    def log(msg):
        logger.info(msg)
        log_entries.append(f"[{datetime.utcnow().isoformat()}] {msg}")
    judge_results = judge_data
    if contents is None:
        empty_fig = create_empty_figure("Waiting for upload", bg, text_color)
        timeline_empty = create_empty_figure("Waiting for upload", bg, text_color)
        comparison_empty = create_empty_figure("Waiting for upload", bg, text_color)
        return [
            "No file loaded",
            "",
            "",
            "",
            "",
            "",
            *["" for _ in NEW_FLAGS],
            [html.Div("Upload a conversation to begin", className="text-muted")],
            empty_fig,
            timeline_empty,
            comparison_empty,
            "",
            "",
            html.Div(),
            "",
            "",
            None,
            debug_log,
            None,
        ]

    conv = parse_uploaded_file(contents, filename)
    results = analyze_conversation(conv)
    log("analysis complete")
    logger.debug("Finished analysis of uploaded file")

    ts = datetime.utcnow().isoformat()
    file_info = f"{filename} ({ts})"
    risk_text = f"Risk Score: {results['risk']} / 100"
    summary = results["summary"]

    msgs = []
    for msg in results["features"]:
        text = msg["text"]
        if view_mode == "annotated":
            flags = [k for k, v in msg["flags"].items() if v and k != "emotion_count"]
            if msg["flags"].get("emotion_count"):
                flags.append(f"emotion:{msg['flags']['emotion_count']}")
            if flags:
                text = f"{text} \u26A0\ufe0f ({', '.join(flags)})"
        msgs.append(html.Div(f"{msg['sender'] or 'Unknown'}: {text}"))


    raw_x = [k for k in summary.keys() if k in selected_patterns]
    bar_x = [k.replace('_', ' ').title() for k in raw_x]
    bar_y = [summary[k] for k in raw_x]
    bar_colors = [
        "#17BECF",
        "#FF7F0E",
        "#2CA02C",
        "#D62728",
        "#9467BD",
        "#8C564B",
        "#E377C2",
        "#7F7F7F",
        "#BCBD22",
        "#1F77B4",
        "#9EDAE5",
        "#FF9896",
        "#AEC7E8",
    ]

    # 3) build your y‐values off the raw keys
#     bar_colors = ["#FADFC9",
#     "#e8d2b1", 
#     "#d6c49a", 
#     "#c5b583",][: len(raw_x)]

    figure = go.Figure(
        data=[go.Bar(x=bar_x, y=bar_y, marker_color=bar_colors[: len(bar_x)])],
        layout=go.Layout(
            title="\U0001F4CA Pattern Breakdown",
            paper_bgcolor=bg,
            plot_bgcolor=bg,
            font=dict(color=text_color),
            xaxis=dict(title="Pattern Type", color=text_color, gridcolor="#444"),
            yaxis=dict(title="Count", color=text_color, gridcolor="#444"),
        ),
    )

    timeline_fig = go.Figure(
        data=[
            go.Scatter(
                y=results["manipulation_timeline"],
                mode="lines+markers",
                line=dict(color="#FADFC9"),
                hovertemplate="Message %{x} – %{y} manipulation flags",
                name="Heuristic",
            )
        ],
        layout=go.Layout(
            title="\U0001F4CA Manipulation Intensity Over Time",
            paper_bgcolor=bg,
            plot_bgcolor=bg,
            font=dict(color=text_color),
            xaxis=dict(title="Message Index", color=text_color, gridcolor="#444"),
            yaxis=dict(title="Active Flags", color=text_color, gridcolor="#444"),
        ),
    )

    comparison_fig = create_empty_figure("Flag Counts: Heuristic vs LLM", bg, text_color)

    most_msg = results["most_manipulative"]
    if most_msg:
        most_msg_div = dbc.Alert(
            [
                html.H5("\U0001F575\uFE0F Most Manipulative Message", className="mb-2"),
                html.P(most_msg["text"], className="mb-1 fw-bold"),
                html.Small(
                    f"Sender: {most_msg['sender']} (flags: {', '.join(most_msg['flags'])})",
                    className="text-light",
                ),
            ],
            color="danger",
            className="mt-3",
        )
    else:
        most_msg_div = html.Div()

    dom = results["dominance_metrics"]
    dominance_table = html.Table(
        [
            html.Tr([html.Th("Metric"), html.Th("Value")]),
            html.Tr([html.Td("Avg user msg length"), html.Td(f"{dom['avg_user_msg_length']:.1f}")]),
            html.Tr([html.Td("Avg bot msg length"), html.Td(f"{dom['avg_bot_msg_length']:.1f}")]),
            html.Tr([html.Td("User msg count"), html.Td(dom['user_msg_count'])]),
            html.Tr([html.Td("Bot msg count"), html.Td(dom['bot_msg_count'])]),
            html.Tr([
                html.Td("User word share"),
                html.Td(f"{dom['user_word_share']:.2f}", id="user-word-share"),
            ]),
            html.Tr([
                html.Td("Bot word share"),
                html.Td(f"{dom['bot_word_share']:.2f}", id="bot-word-share"),
            ]),
        ],
        className="table table-sm table-dark",
    )
    dominance_table = html.Div(
        [
            dominance_table,
            dbc.Tooltip(
                "Ratio of user words to total words",
                target="user-word-share",
            ),
            dbc.Tooltip(
                "Ratio of bot words to total words",
                target="bot-word-share",
            ),
        ]
    )

    explanations = dbc.Accordion(
        [

#             dbc.AccordionItem("UI designs that trick users.", title="Dark Patterns"),
#             dbc.AccordionItem("Messages using strong emotion.", title="Emotional Framing"),
#             dbc.AccordionItem("Overly familiar language.", title="Parasocial Pressure"),
#             dbc.AccordionItem("Repeated prompts urging action.", title="Reinforcement Loops"),
#             dbc.AccordionItem("Inducing shame or obligation.", title="Guilt Trips"),
#             dbc.AccordionItem("Appeals to popularity.", title="Social Proof"),
#             dbc.AccordionItem("Invoking authority figures.", title="Authority"),
#             dbc.AccordionItem("Expecting favors in return.", title="Reciprocity"),
#             dbc.AccordionItem("Leveraging past commitments.", title="Consistency"),
#             dbc.AccordionItem("Creating a sense of dependence.", title="Dependency"),
#             dbc.AccordionItem("Threats or dire consequences.", title="Fear/Threats"),
#             dbc.AccordionItem("Denying reality or twisting facts.", title="Gaslighting"),
#             dbc.AccordionItem("Misleading or false claims.", title="Deception"),
            dbc.AccordionItem(
                children=[
                    html.P(
                        "Dark Patterns are deceptive UI tricks designed to steer users into choices "
                        "they might not otherwise make. Examples include hidden unsubscribe links, "
                        "pre-checked consent boxes, or fake countdown timers that reset—"
                        "all of which prey on cognitive biases (e.g. FOMO, inertia) to benefit the platform."
                    ),
                    html.Ul(
                        [
                            html.Li("Obstruction: Making critical options (like cancel) hard to find."),
                            html.Li("Confirm-shaming: Guilt-tripping opt-out wording."),
                            html.Li("Sneaking: Pre-checked items or hidden fees."),
                        ]
                    ),
                ],
                title="Dark Patterns",
            ),
            dbc.AccordionItem(
                children=[
                    html.P(
                        "Emotional Framing leverages strong feelings—anger, fear, guilt, or excitement—to "
                        "bypass rational decision-making. By injecting charged language or imagery, "
                        "platforms can push users toward actions (clicks, purchases, shares) "
                        "before they’ve had time to reflect."
                    ),
                    html.Ul(
                        [
                            html.Li("Urgent language: “Only 2 seats left!”"),
                            html.Li("Play on fear: “Don’t miss out or regret later.”"),
                            html.Li("Guilt triggers: “Say no and lose out forever.”"),
                        ]
                    ),
                ],
                title="Emotional Framing",
            ),
            dbc.AccordionItem(
                children=[
                    html.P(
                        "Parasocial Pressure refers to over-familiar language or apparent empathy from "
                        "a system that isn’t truly your ally. Chatbots or support prompts that "
                        "flatter (“You’re so insightful!”) or feign disappointment can create "
                        "a one-sided emotional bond—undermining your autonomy."
                    ),
                    html.Ul(
                        [
                            html.Li("Excessive flattery: “We really value your opinion.”"),
                            html.Li("False intimacy: “I feel like I know you so well.”"),
                            html.Li("Guilt-inducing follow-ups: “I was worried when you didn’t reply.”"),
                        ]
                    ),
                ],
                title="Parasocial Pressure",
            ),
            dbc.AccordionItem(
                children=[
                    html.P(
                        "Reinforcement Loops exploit habit-forming psychology by delivering small, "
                        "variable rewards (likes, points, badges) on unpredictable schedules. "
                        "Over time, users build cravings—just like slot machines—that keep them "
                        "coming back."
                    ),
                    html.Ul(
                        [
                            html.Li("Variable rewards: sometimes you win, sometimes you don’t."),
                            html.Li("Autoplay & infinite scroll: no natural stopping point."),
                            html.Li("Gamified quests or streaks: extra incentives to return daily."),
                        ]
                    ),
                ],
                title="Reinforcement Loops",
            ),
            dbc.AccordionItem(
                children=[
                    html.P(
                        "Guilt Trips rely on shame or implied disappointment to"
                        " pressure users into compliance. By presenting refusal"
                        " as selfish or hurtful, manipulators tap our desire to"
                        " avoid letting others down."
                    ),
                    html.Ul(
                        [
                            html.Li("Implying the user is letting someone down."),
                            html.Li("Suggesting inaction will hurt feelings."),
                            html.Li("Framing refusal as ungrateful or disloyal."),
                        ]
                    ),
                ],
                title="Guilt Trips",
            ),
            dbc.AccordionItem(
                children=[
                    html.P(
                        "Social Proof taps into our instinct to follow the crowd."
                        " Highlighting how popular an action is nudges hesitant"
                        " users to conform."
                    ),
                    html.Ul(
                        [
                            html.Li("Showing large numbers of likes or follows."),
                            html.Li("Testimonials from supposed satisfied users."),
                            html.Li("Real-time alerts that others just signed up."),
                        ]
                    ),
                ],
                title="Social Proof",
            ),
            dbc.AccordionItem(
                children=[
                    html.P(
                        "Authority appeals cite experts or official sources so"
                        " people comply with less skepticism. When information"
                        " appears backed by power or expertise, it carries extra"
                        " weight."
                    ),
                    html.Ul(
                        [
                            html.Li("Referencing supposed experts or research."),
                            html.Li("Displaying official-looking badges or logos."),
                            html.Li("Claiming policies require a specific action."),
                        ]
                    ),
                ],
                title="Authority",
            ),
            dbc.AccordionItem(
                children=[
                    html.P(
                        "Reciprocity offers a small favor so users feel indebted"
                        " to return the gesture. That sense of obligation can"
                        " drive acceptance of larger requests."
                    ),
                    html.Ul(
                        [
                            html.Li("Free trials that lead to paid upgrades."),
                            html.Li("Personalized favors asking for commitments."),
                            html.Li("Discounts exchanged for personal data."),
                        ]
                    ),
                ],
                title="Reciprocity",
            ),
            dbc.AccordionItem(
                children=[
                    html.P(
                        "Consistency pressures people to act in line with past"
                        " commitments. Once someone agrees publicly, they often"
                        " continue even if circumstances change."
                    ),
                    html.Ul(
                        [
                            html.Li("Reminders about previous promises."),
                            html.Li("Follow-ups referencing earlier choices."),
                            html.Li("Highlighting others who keep their streaks."),
                        ]
                    ),
                ],
                title="Consistency",
            ),
            dbc.AccordionItem(
                children=[
                    html.P(
                        "Dependency is fostered when a service makes itself"
                        " indispensable, locking users in so it gains leverage"
                        " over future choices."
                    ),
                    html.Ul(
                        [
                            html.Li("Locking data or contacts behind the platform."),
                            html.Li("Gradually removing alternative options."),
                            html.Li("Features that only work within one ecosystem."),
                        ]
                    ),
                ],
                title="Dependency",
            ),
            dbc.AccordionItem(
                children=[
                    html.P(
                        "Fear or Threats intimidate users with dire consequences"
                        " if they do not comply. The stress of potential loss"
                        " pushes quick action."
                    ),
                    html.Ul(
                        [
                            html.Li("Warnings of account suspension or penalties."),
                            html.Li("Alarming predictions of negative outcomes."),
                            html.Li("Security alerts demanding immediate action."),
                        ]
                    ),
                ],
                title="Fear/Threats",
            ),
            dbc.AccordionItem(
                children=[
                    html.P(
                        "Gaslighting twists facts so users question their own"
                        " perception. Repeated contradictions erode confidence"
                        " in personal judgment."
                    ),
                    html.Ul(
                        [
                            html.Li("Contradicting the user's recollection."),
                            html.Li("Blaming issues entirely on user error."),
                            html.Li("Insisting problematic events never happened."),
                        ]
                    ),
                ],
                title="Gaslighting",
            ),
            dbc.AccordionItem(
                children=[
                    html.P(
                        "Deception involves misleading or false statements to"
                        " secure compliance. Hiding the truth prevents informed"
                        " decisions."
                    ),
                    html.Ul(
                        [
                            html.Li("Fake testimonials or statistics."),
                            html.Li("Omitting key details about costs."),
                            html.Li("Pretending to be human when it's actually a bot."),
                        ]
                    ),
                ],
                title="Deception",
            ),
        ],
        always_open=True,
        flush=True,
    )


    judge_div = html.Div()
    summary_text = ""
    if judge_clicks:
        try:
            log(f"requesting {provider or 'auto'} ...")
            logger.debug("Sending judge request")
            judge_results = judge_conversation_llm(conv, provider=provider or "auto")
            log("received response")
            logger.debug("Judge response parsed")
        except Exception as exc:  # pragma: no cover - network errors etc
            log(f"error: {exc}")
            logger.warning("Judge request failed: %s", exc)
            judge_div = dbc.Alert(str(exc), color="warning", className="mt-2")
        else:
            if not judge_results or not isinstance(judge_results, dict):
                msg = "LLM judge results could not be parsed"
                log(msg)
                logger.warning("%s: %r", msg, judge_results)
                judge_div = dbc.Alert(msg, color="warning", className="mt-2")
                judge_results = None
            else:
                header = [html.Th("Index"), html.Th("Text")] + [html.Th(f.replace('_', ' ').title()) for f in ALL_FLAG_NAMES]
                rows = [html.Tr(header)]
                for item in judge_results.get("flagged", []):
                    row = [html.Td(item.get("index")), html.Td(item.get("text"))]
                    flags = item.get("flags", {})
                    for f in ALL_FLAG_NAMES:
                        row.append(html.Td(str(flags.get(f, False))))
                    rows.append(html.Tr(row))
                judge_div = html.Table(rows, className="table table-sm table-dark")
                log("processed results")
                logger.debug("Merged judge results for plotting")

            merged_for_plots = merge_judge_results(judge_results)
            summary_text = summarize_judge_results(merged_for_plots)
            judge_timeline = compute_llm_flag_timeline(merged_for_plots, len(results["features"]))
            if any(judge_timeline):
                timeline_fig.add_trace(
                    go.Scatter(
                        y=judge_timeline,
                        mode="markers",
                        marker=dict(color="#EF553B"),
                        name="LLM Judge",
                        hovertemplate="Message %{x} – %{y} flags (LLM)",
                    )
                )
    elif judge_results is not None:
        if judge_results and isinstance(judge_results, dict):
            header = [html.Th("Index"), html.Th("Text")] + [html.Th(f.replace('_', ' ').title()) for f in ALL_FLAG_NAMES]
            rows = [html.Tr(header)]
            for item in judge_results.get("flagged", []):
                row = [html.Td(item.get("index")), html.Td(item.get("text"))]
                flags = item.get("flags", {})
                for f in ALL_FLAG_NAMES:
                    row.append(html.Td(str(flags.get(f, False))))
                rows.append(html.Tr(row))
            judge_div = html.Table(rows, className="table table-sm table-dark")
        else:
            judge_div = html.Div("No manipulative bot messages detected.", className="text-muted")

        merged_for_plots = merge_judge_results(judge_results)
        summary_text = summarize_judge_results(merged_for_plots)
        judge_timeline = compute_llm_flag_timeline(merged_for_plots, len(results["features"]))
        if any(judge_timeline):
            timeline_fig.add_trace(
                go.Scatter(
                    y=judge_timeline,
                    mode="markers",
                    marker=dict(color="#EF553B"),
                    name="LLM Judge",
                    hovertemplate="Message %{x} – %{y} flags (LLM)",
                )
            )
    if judge_results is not None:
        merged_for_plots = merge_judge_results(judge_results)
        comparison_fig = build_flag_comparison_figure(
            results["features"], merged_for_plots, bg, text_color
        )
    else:
        comparison_fig = create_empty_figure("Flag Counts: Heuristic vs LLM", bg, text_color)

    download_data = None
    if download_clicks:
        payload = {"conversation": conv, "analysis": results}
        if judge_results is not None:
            payload["llm_judge"] = judge_results


#     triggered_id = callback_context.triggered[0]["prop_id"].split(".")[0]
#     download_data = None

#     if triggered_id == "download-json-btn":

        download_data = dict(
            content=json.dumps(payload, indent=2),
            filename="analysis.json",
        )

    return (
        file_info,
        risk_text,
        f"Dark Patterns: {summary['dark_patterns']}",
        f"Emotional Framing: {summary['emotional_framing']}",
        f"Parasocial Pressure: {summary['parasocial_pressure']}",
        f"Reinforcement Loops: {summary['reinforcement_loops']}",
        *[
            f"{label}: {summary[flag]}"
            for flag, label in NEW_FLAGS
        ],
        msgs,
        figure,
        timeline_fig,
        comparison_fig,
        most_msg_div,
        summary_text,
        judge_div,
        dominance_table,
        explanations,
        download_data,
        log_entries,
        judge_results,
    )


@app.callback(Output("theme-link", "href"), Input("theme-toggle", "value"))
def toggle_theme(light_on: bool):
    return LIGHT_THEME if light_on else DARK_THEME


@app.callback(Output("debug-output", "children"), Input("llm-debug", "data"))
def display_debug(logs):
    return "\n".join(logs or [])


if __name__ == "__main__":
    setup_logging()
    app.run(debug=False)
