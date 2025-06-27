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

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

import dash_bootstrap_components as dbc

# pick one of the Bootswatch themes below:
# ['CERULEAN','COSMO','CYBORG','DARKLY','FLATLY','JOURNAL',
#  'LUMEN','PULSE','SLATE','SOLAR','SPACELAB',
#  'SUPERHERO','UNITED','VAPOR','YETI']

from scripts import input_parser, static_feature_extractor
import scorer


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


external_stylesheets = [dbc.themes.DARKLY]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Alethia Manipulation Transparency Console"

default_figure = go.Figure(
    data=[go.Bar(x=[], y=[], marker_color="#17BECF")],
    layout=go.Layout(
        title="Pattern Breakdown",
        paper_bgcolor="#1a1a1a",
        plot_bgcolor="#1a1a1a",
        font=dict(color="white"),
        xaxis=dict(title="Pattern Type", color="white"),
        yaxis=dict(title="Count", color="white"),
    ),
)


app.layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row(
            dbc.Col(
                html.H1(
                    "Alethia Manipulation Transparency Console",
                    className="text-center text-light my-4",
                )
            ),
            justify="center",
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
                                        ],
                                        value=[
                                            "dark_patterns",
                                            "emotional_framing",
                                            "parasocial_pressure",
                                            "reinforcement_loops",
                                        ],
                                        inline=False,
                                        className="mb-3 text-light",
                                    ),
                                    html.Div(id="file-info", className="text-muted mb-2"),
                                    html.H5("Manipulation Risk", className="text-light"),
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
                                        ],
                                        flush=True,
                                    ),
                                ]
                            ),
                        ],
                        className="h-100",
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
                                    dbc.Button(
                                        "Download JSON Report",
                                        id="download-json-btn",
                                        color="secondary",
                                        className="mt-3",
                                    ),
                                    dcc.Download(id="download-json"),
                                ]
                            ),
                        ]
                    ),
                    width=6,
                ),
                # Explanations & Metrics
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("Explanations & Examples"),
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
                        ]
                    ),
                    width=3,
                ),
            ],
            className="g-4 mb-4",
        ),
    ],
)


@app.callback(
    [
        Output("file-info", "children"),
        Output("risk-score", "children"),
        Output("dark-patterns", "children"),
        Output("emotional-framing", "children"),
        Output("parasocial-pressure", "children"),
        Output("reinforcement-loops", "children"),
        Output("conversation-view", "children"),
        Output("pattern-graph", "figure"),
        Output("manipulation-graph", "figure"),
        Output("most-manipulative", "children"),
        Output("dominance-table", "children"),
        Output("explanations", "children"),
        Output("download-json", "data"),
    ],
    [
        Input("upload-data", "contents"),
        Input("view-mode", "value"),
        Input("download-json-btn", "n_clicks"),
        Input("pattern-filter", "value"),
    ],
    [State("upload-data", "filename")],
)
def update_output(contents, view_mode, download_clicks, selected_patterns, filename):
    if contents is None:
        empty_fig = go.Figure()
        return ["", "", "", "", "", "", [], empty_fig, empty_fig, "", "", "", None]

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
    figure = go.Figure(
        data=[go.Bar(x=bar_x, y=bar_y, marker_color="#fadfc9")],
        layout=go.Layout(
            title="Pattern Breakdown",
            paper_bgcolor="#1a1a1a",
            plot_bgcolor="#1a1a1a",
            font=dict(color="white"),
            xaxis=dict(title="Pattern Type", color="white"),
            yaxis=dict(title="Count", color="white"),
        ),
    )

    timeline_fig = go.Figure(
        data=[
            go.Scatter(
                y=results["manipulation_timeline"],
                mode="lines+markers",
                line=dict(color="#ffa15a"),
            )
        ],
        layout=go.Layout(
            title="Manipulation Intensity Over Time",
            paper_bgcolor="#1a1a1a",
            plot_bgcolor="#1a1a1a",
            font=dict(color="white"),
            xaxis=dict(title="Message Index", color="white"),
            yaxis=dict(title="Active Flags", color="white"),
        ),
    )

    most_msg = results["most_manipulative"]
    if most_msg:
        most_msg_div = html.Div(
            [
                html.H5("Most Manipulative Message", className="text-light"),
                html.P(most_msg["text"], className="mb-0"),
                html.Small(
                    f"Sender: {most_msg['sender']} (flags: {', '.join(most_msg['flags'])})",
                    className="text-muted",
                ),
            ]
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
            html.Tr([html.Td("User word share"), html.Td(f"{dom['user_word_share']:.2f}")]),
            html.Tr([html.Td("Bot word share"), html.Td(f"{dom['bot_word_share']:.2f}")]),
        ],
        className="table table-sm table-dark",
    )

    explanations = html.Ul(
        [
            html.Li([html.B("Dark Patterns: "), "UI designs that trick users."]),
            html.Li([html.B("Emotional Framing: "), "Messages using strong emotion."]),
            html.Li([html.B("Parasocial Pressure: "), "Overly familiar language."]),
            html.Li([
                html.B("Reinforcement Loops:"),
                "Repeated prompts urging action.",
            ]),
        ]
    )

    download_data = None
    if download_clicks:
        download_data = dict(
            content=json.dumps({"conversation": conv, "analysis": results}, indent=2),
            filename="analysis.json",
        )

    return (
        file_info,
        risk_text,
        f"Dark Patterns: {summary['dark_patterns']}",
        f"Emotional Framing: {summary['emotional_framing']}",
        f"Parasocial Pressure: {summary['parasocial_pressure']}",
        f"Reinforcement Loops: {summary['reinforcement_loops']}",
        msgs,
        figure,
        timeline_fig,
        most_msg_div,
        dominance_table,
        explanations,
        download_data,
    )


if __name__ == "__main__":
    app.run(debug=False)
