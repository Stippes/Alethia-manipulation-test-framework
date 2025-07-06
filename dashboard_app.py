import base64
import io
import json
from datetime import datetime
from typing import Dict, Any, List

from insight_helpers import (
    compute_manipulation_ratio,
    compute_manipulation_timeline,
    compute_most_manipulative_message,
    compute_dominance_metrics,
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

# pick one of the Bootswatch themes below:
# ['CERULEAN','COSMO','CYBORG','DARKLY','FLATLY','JOURNAL',
#  'LUMEN','PULSE','SLATE','SOLAR','SPACELAB',
#  'SUPERHERO','UNITED','VAPOR','YETI']

from scripts import input_parser, static_feature_extractor
from scripts.judge_conversation import judge_conversation_llm
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


def parse_uploaded_file(contents: str, filename: str) -> Dict[str, Any]:
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
    features = static_feature_extractor.extract_conversation_features(conv)
    trust_score = scorer.score_trust(features)
    risk = round((1.0 - trust_score) * 1000)
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

    return {
        'features': features,
        'risk': risk,
        'summary': summary,
        'manipulation_ratio': manipulation_ratio,
        'manipulation_timeline': manipulation_timeline,
        'most_manipulative': most_manipulative,
        'dominance_metrics': dominance_metrics,
    }


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


app.layout = html.Div([
    html.Link(id="theme-link", rel="stylesheet", href=DARK_THEME),
    dcc.Store(id="theme-store", data="dark"),
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
                                            {"label": "OpenAI", "value": "openai"},
                                            {"label": "Claude", "value": "claude"},
                                            {"label": "Mistral", "value": "mistral"},
                                            {"label": "Gemini", "value": "gemini"},
                                        ],
                                        value="openai",
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
                                    html.Div(id="most-manipulative", className="mt-3 text-light"),
                                    html.Div(id="llm-judge-results", className="mt-3"),
                                    dbc.Button(
                                        "Download JSON Report",
                                        id="download-json-btn",
                                        color="secondary",
                                        className="mt-3",
                                    ),
                                    dcc.Download(id="download-json"),
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
        Output("most-manipulative", "children"),
        Output("llm-judge-results", "children"),
        Output("dominance-table", "children"),
        Output("explanations", "children"),
        Output("download-json", "data"),
    ],
    [
        Input("upload-data", "contents"),
        Input("view-mode", "value"),
        Input("download-json-btn", "n_clicks"),
        Input("llm-judge-btn", "n_clicks"),
        Input("llm-provider", "value"),
        Input("pattern-filter", "value"),
    ],
    [State("upload-data", "filename"), State("theme-toggle", "value")],
)
def update_output(contents, view_mode, download_clicks, judge_clicks, provider, selected_patterns, filename, light_on):
    bg = "#ffffff" if light_on else "#1a1a1a"
    text_color = "black" if light_on else "white"
    if contents is None:
        empty_fig = go.Figure(layout=go.Layout(paper_bgcolor=bg, plot_bgcolor=bg))
        return [
            "",
            "",
            "",
            "",
            "",
            "",
            *["" for _ in NEW_FLAGS],
            [],
            empty_fig,
            empty_fig,
            "",
            html.Div(),
            "",
            "",
            None,
        ]

    conv = parse_uploaded_file(contents, filename)
    results = analyze_conversation(conv)

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

    bar_x = [k for k in summary.keys() if k in selected_patterns]
    bar_y = [summary[k] for k in bar_x]
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
                line=dict(color="#ffa15a"),
                hovertemplate="Message %{x} – %{y} manipulation flags",
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
            dbc.AccordionItem("UI designs that trick users.", title="Dark Patterns"),
            dbc.AccordionItem("Messages using strong emotion.", title="Emotional Framing"),
            dbc.AccordionItem("Overly familiar language.", title="Parasocial Pressure"),
            dbc.AccordionItem("Repeated prompts urging action.", title="Reinforcement Loops"),
            dbc.AccordionItem("Inducing shame or obligation.", title="Guilt Trips"),
            dbc.AccordionItem("Appeals to popularity.", title="Social Proof"),
            dbc.AccordionItem("Invoking authority figures.", title="Authority"),
            dbc.AccordionItem("Expecting favors in return.", title="Reciprocity"),
            dbc.AccordionItem("Leveraging past commitments.", title="Consistency"),
            dbc.AccordionItem("Creating a sense of dependence.", title="Dependency"),
            dbc.AccordionItem("Threats or dire consequences.", title="Fear/Threats"),
            dbc.AccordionItem("Denying reality or twisting facts.", title="Gaslighting"),
            dbc.AccordionItem("Misleading or false claims.", title="Deception"),
        ],
        always_open=True,
        flush=True,
    )

    judge_results = None
    judge_div = html.Div()
    if judge_clicks:
        try:
            judge_results = judge_conversation_llm(conv, provider=provider or "openai")
        except Exception as exc:  # pragma: no cover - network errors etc
            judge_div = dbc.Alert(str(exc), color="warning", className="mt-2")
        else:
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

    download_data = None
    if download_clicks:
        payload = {"conversation": conv, "analysis": results}
        if judge_results is not None:
            payload["llm_judge"] = judge_results
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
        most_msg_div,
        judge_div,
        dominance_table,
        explanations,
        download_data,
    )


@app.callback(Output("theme-link", "href"), Input("theme-toggle", "value"))
def toggle_theme(light_on: bool):
    return LIGHT_THEME if light_on else DARK_THEME


if __name__ == "__main__":
    app.run(debug=False)
