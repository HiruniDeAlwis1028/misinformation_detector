from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class MisinformationDetector:
    def __init__(self, model_path="./trained-model-pro"):
        self.model_path = model_path
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            print("✅ Misinformation Detector Loaded!")
        except:
            print("❌ Please train the model first (run 03_train_model_pro.py)")
            return
    
    def detect_misinformation(self, text, behavior='expert'):
        """Detect misinformation in text"""
        try:
            # Behavior-specific prompts
            behavior_prompts = {
                'expert': "Provide a technical, detailed analysis:",
                'casual': "Give a simple, friendly explanation:", 
                'business': "Offer a professional, business-focused response:"
            }
            
            prompt = f"User: Analyze this information: {text}\nAssistant: {behavior_prompts.get(behavior, behavior_prompts['expert'])}"
            
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=400,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant response
            if "Assistant:" in response:
                answer = response.split("Assistant:")[-1].strip()
            else:
                answer = response
            
            # Detect misinformation
            misinfo_indicators = ['❌ MISINFORMATION', '❌ FALSE', '❌ INACCURATE', 'debunked', 'false', 'not true']
            factual_indicators = ['✅ FACTUAL', '✅ TRUE', '✅ ACCURATE', 'verified', 'true', 'accurate', 'supported']
            
            answer_lower = answer.lower()
            is_misinformation = any(indicator in answer_lower for indicator in misinfo_indicators)
            is_factual = any(indicator in answer_lower for indicator in factual_indicators)
            
            return {
                'input_text': text,
                'analysis': answer,
                'contains_misinformation': is_misinformation,
                'is_factual_information': is_factual,
                'behavior_style': behavior
            }
            
        except Exception as e:
            return {
                'input_text': text,
                'analysis': f"Error: {str(e)}",
                'contains_misinformation': None,
                'is_factual_information': None,
                'behavior_style': behavior
            }

def interactive_chat():
    """Interactive chat with the trained model"""
    print(" MISINFORMATION DETECTOR CHATBOT")
    print("=" * 50)
    
    detector = MisinformationDetector()
    if not hasattr(detector, 'model'):
        return
    
    print("Available behaviors: expert, casual, business")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            behavior = input("Choose behavior (expert/casual/business): ").strip().lower()
            if behavior == 'quit':
                break
            if behavior not in ['expert', 'casual', 'business']:
                print("Using 'expert' as default")
                behavior = 'expert'
            
            user_input = input("\n Your text: ").strip()
            if user_input.lower() in ['quit', 'exit']:
                break
            if not user_input:
                continue
            
            print(" Analyzing...")
            result = detector.detect_misinformation(user_input, behavior)
            
            print("\n" + "=" * 50)
            if result['contains_misinformation']:
                print(" MISINFORMATION DETECTED")
            elif result['is_factual_information']:
                print(" FACTUAL INFORMATION")
            else:
                print(" UNCERTAIN - Needs verification")
            
            print(f" Behavior: {result['behavior_style'].upper()}")
            print(f" Analysis: {result['analysis']}")
            print("=" * 50)
            print()
            
        except KeyboardInterrupt:
            print("\n Goodbye!")
            break
        except Exception as e:
            print(f" Error: {e}")

if __name__ == "__main__":
    interactive_chat()
