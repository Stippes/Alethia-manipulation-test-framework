import base64
import io
import json
from datetime import datetime
from typing import Dict, Any, List

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

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
    risk = round((1.0 - trust_score) * 100)
    summary = {
        'dark_patterns': sum(1 for f in features if f['flags'].get('dark_ui')),
        'emotional_framing': sum(f['flags'].get('emotion_count', 0) for f in features),
        'parasocial_pressure': sum(1 for f in features if f['flags'].get('flattery')),
        'reinforcement_loops': sum(1 for f in features if f['flags'].get('urgency') or f['flags'].get('fomo')),
    }
    return {'features': features, 'risk': risk, 'summary': summary}


app = dash.Dash(__name__)
app.title = "Alethia Manipulation Transparency Console"

app.layout = html.Div([
    html.H1("Alethia Manipulation Transparency Console"),
    html.Div([
        html.Div([
            dcc.Upload(id='upload-data', children=html.Button('Upload Conversation'), multiple=False),
            html.Br(),
            dcc.Dropdown(
                id='conv-type',
                options=[{'label': 'Chatbot', 'value': 'chatbot'}, {'label': 'Social Media', 'value': 'social'}],
                value='chatbot'
            ),
            html.Div(id='file-info'),
            html.H3("Manipulation Risk"),
            html.Div(id='risk-score', style={'fontSize': '24px', 'fontWeight': 'bold'}),
            html.Ul([
                html.Li(id='dark-patterns'),
                html.Li(id='emotional-framing'),
                html.Li(id='parasocial-pressure'),
                html.Li(id='reinforcement-loops'),
            ])
        ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px', 'boxSizing': 'border-box'}),
        html.Div([
            html.Div([
                dcc.RadioItems(
                    id='view-mode',
                    options=[{'label': 'Clean', 'value': 'clean'}, {'label': 'Annotated', 'value': 'annotated'}],
                    value='clean', inline=True
                )
            ]),
            html.Div(id='conversation-view', style={'height': '400px', 'overflowY': 'scroll', 'border': '1px solid #ccc', 'padding': '5px'}),
            dcc.Graph(id='pattern-graph'),
            html.Button('Download JSON Report', id='download-json-btn'),
            dcc.Download(id='download-json'),
        ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px', 'boxSizing': 'border-box'}),
        html.Div([
            html.H3("Explanations & Examples"),
            html.Div(id='explanations'),
            html.H3("What You Can Do"),
            html.Ul([
                html.Li("Turn off autoplay / limit notifications"),
                html.Li("Save conversation logs"),
                html.Li("Disable tracking settings"),
            ])
        ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px', 'boxSizing': 'border-box'})
    ])
])

@app.callback(
    [Output('file-info', 'children'),
     Output('risk-score', 'children'),
     Output('dark-patterns', 'children'),
     Output('emotional-framing', 'children'),
     Output('parasocial-pressure', 'children'),
     Output('reinforcement-loops', 'children'),
     Output('conversation-view', 'children'),
     Output('pattern-graph', 'figure'),
     Output('explanations', 'children'),
     Output('download-json', 'data')],
    [Input('upload-data', 'contents'), Input('view-mode', 'value'), Input('download-json-btn', 'n_clicks')],
    [State('upload-data', 'filename')]
)
def update_output(contents, view_mode, download_clicks, filename):
    if contents is None:
        return ['', '', '', '', '', '', '', go.Figure(), '', None]

    conv = parse_uploaded_file(contents, filename)
    results = analyze_conversation(conv)
    ts = datetime.utcnow().isoformat()
    file_info = f"{filename} ({ts})"
    risk_text = f"Risk Score: {results['risk']} / 100"
    summary = results['summary']

    msgs = []
    for msg in results['features']:
        text = msg['text']
        if view_mode == 'annotated':
            flags = [k for k, v in msg['flags'].items() if v and k != 'emotion_count']
            if msg['flags'].get('emotion_count'):
                flags.append(f"emotion:{msg['flags']['emotion_count']}")
            if flags:
                text = f"{text} \u26A0\ufe0f ({', '.join(flags)})"
        msgs.append(html.Div(f"{msg['sender'] or 'Unknown'}: {text}"))

    figure = go.Figure(
        data=[go.Bar(x=list(summary.keys()), y=list(summary.values()))],
        layout=go.Layout(title='Pattern Breakdown')
    )

    explanations = html.Ul([
        html.Li(html.B('Dark Patterns: ') + 'UI designs that trick users.'),
        html.Li(html.B('Emotional Framing: ') + 'Messages using strong emotion.'),
        html.Li(html.B('Parasocial Pressure: ') + 'Overly familiar language.'),
        html.Li(html.B('Reinforcement Loops: ') + 'Repeated prompts urging action.')
    ])

    download_data = None
    if download_clicks:
        download_data = dict(content=json.dumps({'conversation': conv, 'analysis': results}, indent=2), filename='analysis.json')

    return (
        file_info,
        risk_text,
        f"Dark Patterns: {summary['dark_patterns']}",
        f"Emotional Framing: {summary['emotional_framing']}",
        f"Parasocial Pressure: {summary['parasocial_pressure']}",
        f"Reinforcement Loops: {summary['reinforcement_loops']}",
        msgs,
        figure,
        explanations,
        download_data
    )


if __name__ == '__main__':
    app.run_server(debug=False)