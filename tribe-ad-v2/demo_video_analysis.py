#!/usr/bin/env python3
"""
NeuroAD V2 - Video Processing Demo
==================================

This script demonstrates the complete pipeline:
Video Input → TRIBE v2 → 12 Cognitive Signals
"""

import json
import sys
import time
from pathlib import Path

def test_video_processing():
    print("🎬 NeuroAD V2 - Video to 12 Cognitive Signals Demo")
    print("=" * 60)
    
    # Change to project directory
    project_dir = Path("c:/Users/Lenovo/Desktop/NeuroAD/tribe-ad-v2")
    import os
    os.chdir(project_dir)
    sys.path.insert(0, str(project_dir))
    
    try:
        # Import the services
        from app.services.video import extract_frames
        from app.services.tribe import get_brain_predictions
        from app.services.signals import compute_signals
        from app.services.patterns import classify_pattern
        from app.services.insights import generate_insights
        
        print("1️⃣ Loading test video...")
        video_path = project_dir / "test_video.mp4"
        if not video_path.exists():
            print(f"❌ Test video not found at {video_path}")
            return
        
        print(f"   📹 Video: {video_path.name} ({video_path.stat().st_size} bytes)")
        
        print("\n2️⃣ Extracting frames at 1 FPS...")
        frames = extract_frames(str(video_path))
        print(f"   🖼️  Extracted {len(frames)} frames")
        
        if not frames:
            print("   ❌ No frames could be extracted")
            return
        
        print("\n3️⃣ Running TRIBE v2 brain model...")
        start_time = time.perf_counter()
        brain_vectors = get_brain_predictions(frames)
        inference_time = time.perf_counter() - start_time
        
        print(f"   🧠 Generated {len(brain_vectors)} brain vectors")
        print(f"   📊 Each vector: {brain_vectors[0].shape} dimensions")
        print(f"   ⏱️  Inference time: {inference_time:.2f} seconds")
        
        print("\n4️⃣ Computing 12 cognitive signals...")
        signals = compute_signals(brain_vectors)
        
        print("   🧬 Cognitive Signals (0-10 scale):")
        print("   " + "=" * 45)
        
        # Group by category
        categories = {
            "Attention": ["saliency", "motion", "novelty"],
            "Emotion": ["emotion_intensity", "emotion_valence", "relatability"],
            "Memory": ["distinctiveness", "repetition", "story_coherence"],  
            "Cognitive": ["cognitive_load", "clarity", "info_density"]
        }
        
        for category, signal_names in categories.items():
            print(f"   📋 {category}:")
            for signal in signal_names:
                if signal in signals:
                    value = signals[signal]
                    bar = "█" * int(value) + "░" * (10 - int(value))
                    print(f"      {signal:<18}: {value:>4.1f} |{bar}|")
        
        print("\n5️⃣ Pattern classification...")
        cluster = classify_pattern(signals)
        cluster_names = {
            0: "High-energy, emotional content",
            1: "Balanced professional style", 
            2: "Story-driven narrative",
            3: "Information-heavy explainer",
            4: "Premium high-impact",
            5: "Simple low-complexity"
        }
        
        print(f"   🎯 Pattern Cluster: {cluster} ({cluster_names.get(cluster, 'Unknown')})")
        
        print("\n6️⃣ Generating insights...")
        insights = generate_insights(signals, cluster)
        overall_score = sum(signals.values()) / len(signals)
        
        print(f"   📊 Overall Score: {overall_score:.1f}/10")
        print("   💡 Insights:")
        for insight in insights:
            print(f"      • {insight}")
        
        print(f"\n🎉 SUCCESS! Complete analysis in {time.perf_counter() - start_time + inference_time:.2f} seconds")
        
        # Show API format
        result = {
            "signals": signals,
            "cluster": cluster,
            "insights": insights,
            "timing_seconds": round(inference_time, 4),
            "overall_score": round(overall_score, 3)
        }
        
        print(f"\n📄 Complete API Response:")
        print(json.dumps(result, indent=2))
        
        return True
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_video_processing()
    if success:
        print("\n✅ NeuroAD V2 is working perfectly!")
        print("\n🚀 To run the web server:")
        print("   cd c:\\Users\\Lenovo\\Desktop\\NeuroAD\\tribe-ad-v2")
        print("   uvicorn app.main:app --reload")
        print("\n📹 To process your own videos:")
        print("   curl -X POST \"http://127.0.0.1:8000/analyze\" -F \"file=@your_video.mp4\"")
    else:
        print("\n❌ There was an issue. Check the error messages above.")