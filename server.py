from flask import Flask, request, jsonify
from flask_cors import CORS

# Google Gemini
from google import genai
genai_client = genai.Client(api_key="AIzaSyCSZUI-z4ogIKeB7uDgy-WmHXDPJ0Prf-8")

# Groq models
from groq import Groq
groq_client = Groq(api_key="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")

app = Flask(__name__)
CORS(app)

def get_prompt_from_request():
    data = request.get_json()
    if not data or 'prompt' not in data:
        return None, jsonify({'error': 'Missing "prompt" in request body'}), 400
    return data['prompt'], None, None

@app.route('/gemini', methods=['POST'])
def gemini():
    prompt, error_response, status = get_prompt_from_request()
    if error_response:
        return error_response, status
    try:
        response = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt]
        )
        return jsonify({'response': response.text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/deepseek', methods=['POST'])
def deepseek():
    prompt, error_response, status = get_prompt_from_request()
    if error_response:
        return error_response, status
    try:
        completion = groq_client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_completion_tokens=32768,
            top_p=1,
            stop=None,
        )
        return jsonify({'response': completion.choices[0].message.content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/llama3', methods=['POST'])
def llama3():
    prompt, error_response, status = get_prompt_from_request()
    if error_response:
        return error_response, status
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_completion_tokens=32768,
            top_p=1,
            stop=None,
        )
        return jsonify({'response': completion.choices[0].message.content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/llama4', methods=['POST'])
def llama4():
    prompt, error_response, status = get_prompt_from_request()
    if error_response:
        return error_response, status
    try:
        completion = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_completion_tokens=8192,
            top_p=1,
            stop=None,
        )
        return jsonify({'response': completion.choices[0].message.content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
