from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__)

df = pd.read_csv("/tmp/result_table.csv", index_col=0)

# Function to format code columns
def format_code_columns(df):
    for col in ['python code', 'instructions']:
        if col == 'python code':
            lang = 'python'
        else:
            lang = 'text'
        df[col] = df[col].apply(lambda x: f'<pre><code class="hljs language-{lang}">{x.replace("\n", "&#010;")}</code></pre>')
    return df

# Route to render the table
@app.route('/')
def index():
    stages = df['stage'].unique().tolist()
    return render_template('index.html', stages=stages)

# Route to update the table
@app.route('/update_table', methods=['POST'])
def update_table():
    stage = request.form.get('stage')
    filtered_df = df[df['stage'] == stage].copy()
    formatted_df = format_code_columns(filtered_df[['python code', 'instructions']])
    return jsonify({'table': formatted_df.to_html(index=False, classes='table table-striped table-dark', escape=False)})



if __name__ == "__main__":
    app.run(debug=True)
