 import logging
from openai import OpenAI
from flask import Flask, request, jsonify, render_template

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#put api key here 
api_key = 'API key goes here'
client = OpenAI(api_key=api_key)

app = Flask(__name__)

#This creates another 
CHAT_LOG_FILE = "chat_history.txt"

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_input = data.get('message')
        
        if not user_input:
            logger.warning("No message provided in request")
            return jsonify({'error': 'No message provided'}), 400
        
        logger.info(f"Received message: {user_input[:50]}...")
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ],
            max_tokens=500
        )
        
        ai_response = response.choices[0].message.content
        logger.info(f"Generated response: {ai_response[:50]}...")
        
        # Save to file (overwrites each time)
        with open(CHAT_LOG_FILE, 'w', encoding='utf-8') as f:
            f.write(f"AI: {ai_response}\n")
        
        return jsonify({'response': ai_response})
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': f'Error: {str(e)}'}), 500

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(debug=True)  
