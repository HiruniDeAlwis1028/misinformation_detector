from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from datetime import datetime
import os
import threading
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class AIChatbot:
    def __init__(self, model_path="./trained-model-pro"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.load_model()
        # Best-effort warmup in background to reduce first-call latency
        try:
            threading.Thread(target=self._warmup, daemon=True).start()
        except Exception:
            pass

    def _warmup(self):
        try:
            sample = "User: hello\nAssistant:"
            inputs = self.tokenizer(sample, return_tensors="pt", max_length=32, truncation=True)
            with torch.no_grad():
                _ = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        except Exception:
            pass
    
    def load_model(self):
        """Load your trained AI model"""
        try:
            logger.info("Loading AI model...")
            # Allow forcing a small public fallback model to ensure fast startup
            force_fallback = str(os.getenv("USE_FALLBACK_MODEL", "")).lower() in ("1", "true", "yes", "on")

            use_local = os.path.isdir(self.model_path) and not force_fallback
            if use_local:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, 
                    local_files_only=True,
                    use_safetensors=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                logger.info("AI model loaded from local path with safetensors.")
            else:
                # Fallback to a small public model so the server can start
                fallback_model = "distilgpt2"
                logger.warning(
                    f"Model directory '{self.model_path}' not found. Falling back to '{fallback_model}'."
                )
                self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                self.model = AutoModelForCausalLM.from_pretrained(fallback_model)

            # Ensure pad token exists to avoid generation errors
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("AI model ready.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e
    
    def generate_response(self, message, behavior_style="expert"):
        """Generate AI response with behavior adaptation"""
        try:
            # Behavior-specific prompts
            behavior_prompts = {
                "expert": "Provide a technical, detailed analysis:",
                "casual": "Give a simple, friendly explanation:",
                "business": "Offer a professional, business-focused response:"
            }
            
            # Match 05_use_chatbot.py prompt style
            prompt = f"User: Analyze this information: {message}\nAssistant: {behavior_prompts.get(behavior_style, behavior_prompts['expert'])}"
            
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    # Mirror 05_use_chatbot.py generation behavior
                    max_length=400,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the assistant's response
            if "Assistant:" in response:
                answer = response.split("Assistant:")[-1].strip()
            else:
                answer = response
            
            # Fallback: infer labels conservatively as None unless clearly stated by the model
            is_misinformation = None
            is_factual = None
            
            return {
                "success": True,
                "response": answer,
                "is_misinformation": is_misinformation,
                "is_factual": is_factual,
                "behavior_style": behavior_style,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "Sorry, I encountered an error processing your request."
            }

# Initialize AI chatbot
try:
    chatbot = AIChatbot()
    AI_READY = True
    logger.info("AI Chatbot initialized and ready.")
except Exception as e:
    chatbot = None
    AI_READY = False
    logger.error(f"Failed to initialize AI: {e}")

def generate_with_timeout(chatbot, message, behavior, timeout_seconds=20):
    """Generate AI response with timeout using threading"""
    result = [None]
    error = [None]
    
    def target():
        try:
            result[0] = chatbot.generate_response(message, behavior)
        except Exception as e:
            error[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        # Thread is still running, timeout occurred
        return {
            "success": True,
            "response": "Analysis timed out. This text appears to be complex and requires more processing time.",
            "is_misinformation": None,
            "is_factual": None,
            "behavior_style": behavior,
            "timestamp": datetime.now().isoformat()
        }
    
    if error[0]:
        raise error[0]
    
    return result[0]

# API Routes
@app.route('/')
def home():
    return jsonify({
        "status": "AI Server Running",
        "ai_ready": AI_READY,
        "message": "Behavior-Aware Misinformation Detector API"
    })

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """Main chat endpoint for browser extension"""
    if not AI_READY:
        return jsonify({
            "success": False,
            "error": "AI model not loaded",
            "response": "AI service is currently unavailable."
        })
    
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        behavior = data.get('behavior', 'expert')
        
        if not message:
            return jsonify({
                "success": False,
                "error": "No message provided",
                "response": "Please provide a message to analyze."
            })
        
        logger.info(f"Received request: {message[:50]}... (Behavior: {behavior})")
        
        # Generate AI response with timeout
        result = generate_with_timeout(chatbot, message, behavior, 22)
        
        logger.info(f"Sent response: Misinfo: {result.get('is_misinformation')}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "response": "Internal server error occurred."
        })

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if AI_READY else "unhealthy",
        "ai_ready": AI_READY,
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("Starting AI Server...")
    print("Server will run at: http://localhost:5000")
    print("Available endpoints:")
    print("   - GET  /              - Server status")
    print("   - POST /api/chat      - Main chat endpoint") 
    print("   - GET  /api/health    - Health check")
    print("\nMake sure your trained model is in: ./trained-model-pro/")

    app.run(host='0.0.0.0', port=5000, debug=False)
