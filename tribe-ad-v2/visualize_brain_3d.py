#!/usr/bin/env python3
"""
NeuroAD V2 - 3D Brain Visualization
====================================
Visualizes TRIBE v2 brain activation vectors on the cortical surface.

Usage:
    python visualize_brain_3d.py                    # Use test_video.mp4
    python visualize_brain_3d.py path/to/video.mp4  # Use custom video
"""

import sys
from pathlib import Path

import numpy as np


def get_brain_data_from_video(video_path: Path, max_frames: int = 3):
    """Extract brain vectors from video using TRIBE v2."""
    print(f"🎬 Processing video: {video_path.name}")
    
    # Import services
    project_dir = Path(__file__).parent
    sys.path.insert(0, str(project_dir))
    
    from app.services.video import stream_sampled_frames
    from app.services.tribe import initialize_tribe_model, get_brain_vector
    
    print("🧠 Loading TRIBE v2 model...")
    initialize_tribe_model()
    
    print(f"🖼️  Extracting frames (max {max_frames})...")
    frames = list(stream_sampled_frames(video_path, sample_fps=1))[:max_frames]
    
    print(f"⚙️  Running brain model on {len(frames)} frames...")
    brain_vectors = []
    for i, frame in enumerate(frames):
        print(f"   Processing frame {i+1}/{len(frames)}...", end="", flush=True)
        bv = get_brain_vector(frame)
        brain_vectors.append(bv)
        print(" ✅")
    
    # Average across frames
    avg_vector = np.mean(brain_vectors, axis=0)
    return avg_vector


def visualize_brain_2d_detailed(brain_vector: np.ndarray, output_path: str = "brain_activation.png"):
    """
    Create a detailed 2D brain activation visualization with cortical mapping.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, FancyBboxPatch
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    
    fig = plt.figure(figsize=(16, 10))
    
    # Split into hemispheres (fsaverage5: 10242 vertices per hemisphere)
    n_hemi = 10242
    lh_data = brain_vector[:n_hemi]
    rh_data = brain_vector[n_hemi:n_hemi*2] if len(brain_vector) > n_hemi else brain_vector[:n_hemi]
    
    # Define brain regions (approximate vertex ranges)
    regions = {
        "Visual (V1/V2)": (0, 1500),
        "Visual (V3/V4)": (1500, 2500),
        "Temporal (Auditory)": (2500, 4000),
        "Temporal (Language)": (4000, 5500),
        "Parietal (Attention)": (5500, 7000),
        "Parietal (Spatial)": (7000, 8000),
        "Frontal (Motor)": (8000, 9000),
        "Frontal (Executive)": (9000, 10242),
    }
    
    # Calculate regional activations
    lh_regions = {name: np.mean(lh_data[start:end]) for name, (start, end) in regions.items()}
    rh_regions = {name: np.mean(rh_data[start:end]) for name, (start, end) in regions.items()}
    
    # ---- Plot 1: Brain schematic with activation ----
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1, 1)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title("Brain Activation Map (Top View)", fontsize=14, fontweight='bold')
    
    # Draw brain outline
    brain_left = Ellipse((-0.5, 0), 0.9, 0.8, color='#f0e6d3', ec='black', lw=2)
    brain_right = Ellipse((0.5, 0), 0.9, 0.8, color='#f0e6d3', ec='black', lw=2)
    ax1.add_patch(brain_left)
    ax1.add_patch(brain_right)
    
    # Color map for activations
    all_activations = list(lh_regions.values()) + list(rh_regions.values())
    vmin, vmax = min(all_activations), max(all_activations)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.hot
    
    # Region positions on brain schematic
    region_positions = {
        "Visual (V1/V2)": [(-0.5, -0.25), (0.5, -0.25)],
        "Visual (V3/V4)": [(-0.6, -0.1), (0.6, -0.1)],
        "Temporal (Auditory)": [(-0.75, 0.1), (0.75, 0.1)],
        "Temporal (Language)": [(-0.7, 0.25), (0.7, 0.25)],
        "Parietal (Attention)": [(-0.4, 0.15), (0.4, 0.15)],
        "Parietal (Spatial)": [(-0.35, 0.3), (0.35, 0.3)],
        "Frontal (Motor)": [(-0.3, 0.4), (0.3, 0.4)],
        "Frontal (Executive)": [(-0.4, 0.5), (0.4, 0.5)],
    }
    
    for region, positions in region_positions.items():
        lh_val = lh_regions[region]
        rh_val = rh_regions[region]
        
        # Left hemisphere
        color_l = cmap(norm(lh_val))
        circle_l = plt.Circle(positions[0], 0.08, color=color_l, ec='black', lw=1)
        ax1.add_patch(circle_l)
        
        # Right hemisphere
        color_r = cmap(norm(rh_val))
        circle_r = plt.Circle(positions[1], 0.08, color=color_r, ec='black', lw=1)
        ax1.add_patch(circle_r)
    
    ax1.text(-0.5, -0.55, "Left", ha='center', fontsize=10)
    ax1.text(0.5, -0.55, "Right", ha='center', fontsize=10)
    ax1.text(0, 0.7, "Front", ha='center', fontsize=10)
    ax1.text(0, -0.45, "Back", ha='center', fontsize=10)
    
    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, shrink=0.6, label='Activation')
    
    # ---- Plot 2: Regional bar chart ----
    ax2 = fig.add_subplot(2, 2, 2)
    
    region_names = list(regions.keys())
    x = np.arange(len(region_names))
    width = 0.35
    
    lh_vals = [lh_regions[r] for r in region_names]
    rh_vals = [rh_regions[r] for r in region_names]
    
    bars1 = ax2.barh(x - width/2, lh_vals, width, label='Left Hemisphere', color='coral')
    bars2 = ax2.barh(x + width/2, rh_vals, width, label='Right Hemisphere', color='steelblue')
    
    ax2.set_yticks(x)
    ax2.set_yticklabels(region_names, fontsize=9)
    ax2.set_xlabel('Mean Activation')
    ax2.set_title('Regional Brain Activation', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # ---- Plot 3: Full vertex heatmap ----
    ax3 = fig.add_subplot(2, 2, 3)
    
    # Create 2D representation of brain surface
    n_cols = 100
    n_rows_lh = len(lh_data) // n_cols
    n_rows_rh = len(rh_data) // n_cols
    
    lh_grid = lh_data[:n_rows_lh * n_cols].reshape(n_rows_lh, n_cols)
    rh_grid = rh_data[:n_rows_rh * n_cols].reshape(n_rows_rh, n_cols)
    
    combined_grid = np.vstack([lh_grid, np.zeros((2, n_cols)), rh_grid])
    
    im = ax3.imshow(combined_grid, aspect='auto', cmap='hot', 
                    vmin=np.percentile(brain_vector, 5), 
                    vmax=np.percentile(brain_vector, 95))
    ax3.set_title('Cortical Vertex Activation Map', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Vertex Position (angular)')
    ax3.set_ylabel('Vertex Index')
    ax3.axhline(y=n_rows_lh, color='white', linestyle='-', lw=2)
    ax3.text(50, n_rows_lh//2, 'Left Hemisphere', ha='center', va='center', 
             color='white', fontsize=12, fontweight='bold')
    ax3.text(50, n_rows_lh + n_rows_rh//2 + 2, 'Right Hemisphere', ha='center', va='center', 
             color='white', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax3, label='Activation')
    
    # ---- Plot 4: Activation distribution + stats ----
    ax4 = fig.add_subplot(2, 2, 4)
    
    ax4.hist(brain_vector, bins=60, color='coral', edgecolor='black', alpha=0.7, density=True)
    ax4.axvline(x=np.mean(brain_vector), color='red', linestyle='--', lw=2, label=f'Mean: {np.mean(brain_vector):.3f}')
    ax4.axvline(x=np.median(brain_vector), color='blue', linestyle='--', lw=2, label=f'Median: {np.median(brain_vector):.3f}')
    
    ax4.set_xlabel('Activation Value')
    ax4.set_ylabel('Density')
    ax4.set_title('Activation Distribution', fontsize=14, fontweight='bold')
    ax4.legend()
    
    # Add stats text
    stats_text = f"Total Vertices: {len(brain_vector):,}\n"
    stats_text += f"Max: {np.max(brain_vector):.3f}\n"
    stats_text += f"Min: {np.min(brain_vector):.3f}\n"
    stats_text += f"Std: {np.std(brain_vector):.3f}"
    ax4.text(0.95, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('🧠 NeuroAD V2 - Brain Activation Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"📊 Saved visualization to: {output_path}")
    plt.show()
    
    return output_path


def main():
    import os
    os.chdir(Path(__file__).parent)
    
    # Determine video path
    if len(sys.argv) > 1:
        video_path = Path(sys.argv[1])
    else:
        video_path = Path("test_video.mp4")
    
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)
    
    # Get brain activation data
    print("=" * 60)
    print("🧠 NeuroAD V2 - Brain Visualization")
    print("=" * 60)
    
    brain_vector = get_brain_data_from_video(video_path, max_frames=2)
    print(f"\n✅ Got brain vector: {brain_vector.shape} dimensions")
    
    # Create visualization
    visualize_brain_2d_detailed(brain_vector, "brain_activation_detailed.png")


if __name__ == "__main__":
    main()
