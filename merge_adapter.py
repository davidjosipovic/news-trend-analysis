#!/usr/bin/env python3
"""
Merge fine-tuned adapter into base RoBERTa model.
This creates a single merged model for faster inference.
"""

import os
import argparse
import torch

def merge_adapter(adapter_path, output_path, base_model="cardiffnlp/twitter-roberta-base-sentiment-latest"):
    """
    Merge adapter into base model and save as standalone model.
    
    Args:
        adapter_path: Path to adapter directory
        output_path: Where to save merged model
        base_model: Base model name
    """
    print("=" * 70)
    print("Adapter Merge Tool")
    print("=" * 70)
    print()
    
    # Check if adapter exists
    if not os.path.exists(adapter_path):
        print(f"❌ Adapter not found: {adapter_path}")
        return False
    
    print(f"📦 Loading adapter from: {adapter_path}")
    print(f"🔧 Base model: {base_model}")
    print(f"💾 Output path: {output_path}")
    print()
    
    try:
        from adapters import AutoAdapterModel
        from transformers import RobertaTokenizerFast
        
        # Load model
        print("Loading base model...")
        model = AutoAdapterModel.from_pretrained(base_model)
        tokenizer = RobertaTokenizerFast.from_pretrained(base_model)
        
        # Load adapter
        print("Loading adapter...")
        adapter_name = "sentiment_adapter"
        model.load_adapter(adapter_path, load_as=adapter_name)
        model.set_active_adapters(adapter_name)
        
        print(f"✅ Adapter loaded: {model.active_adapters}")
        
        # Merge adapter into base model
        print("Merging adapter into base model...")
        model.merge_adapter(adapter_name)
        
        print("✅ Adapter merged successfully!")
        
        # Save merged model
        print(f"Saving merged model to: {output_path}")
        os.makedirs(output_path, exist_ok=True)
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        print("✅ Merged model saved!")
        print()
        
        # Test inference
        print("Testing merged model...")
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        test_texts = [
            "Stock prices soared after excellent earnings report!",
            "The market crashed badly today.",
            "Trading was flat with no significant changes."
        ]
        
        for text in test_texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=200)
            enc = {k: v.to(device) for k, v in enc.items()}
            
            with torch.no_grad():
                outputs = model(**enc)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                pred_id = probs.argmax(dim=-1).item()
                confidence = probs.max().item()
            
            # Map prediction
            labels = ['negative', 'neutral', 'positive']
            if pred_id < len(labels):
                sentiment = labels[pred_id]
            else:
                sentiment = f"LABEL_{pred_id}"
            
            print(f"  '{text[:50]}...'")
            print(f"    → {sentiment} (confidence: {confidence:.4f})")
        
        print()
        print("=" * 70)
        print("✅ Merge Complete!")
        print("=" * 70)
        print()
        print("Next steps:")
        print(f"1. Use merged model in your code:")
        print(f'   model = AutoModel.from_pretrained("{output_path}")')
        print()
        print(f"2. Or update analyze_sentiment.py to use:")
        print(f'   model="{output_path}"')
        print()
        
        return True
        
    except ImportError:
        print("❌ Error: 'adapter-transformers' not installed")
        print("   Install with: pip install adapter-transformers")
        return False
    except Exception as e:
        print(f"❌ Error during merge: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge adapter into base model")
    parser.add_argument(
        '--adapter-path',
        type=str,
        default='./models/sentiment_adapter_best',
        help='Path to adapter directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./models/sentiment_roberta_merged',
        help='Output path for merged model'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default='cardiffnlp/twitter-roberta-base-sentiment-latest',
        help='Base model name'
    )
    
    args = parser.parse_args()
    
    success = merge_adapter(
        adapter_path=args.adapter_path,
        output_path=args.output,
        base_model=args.base_model
    )
    
    exit(0 if success else 1)
