"""
Baseball Vision - Streamlit UI for Video Inference
Run with: streamlit run app.py
"""

import streamlit as st
import os
import requests
import sys
from pathlib import Path
from main import predict, BASE_DIR
import time
from datetime import datetime, timedelta

# Import mlb_clip_downloader from video_downloader folder
VIDEO_DOWNLOADER_DIR = BASE_DIR.parent / "video_downloader"
sys.path.insert(0, str(VIDEO_DOWNLOADER_DIR))
from mlb_clip_downloader import (
    search_mlb_clips,
    download_clip,
    get_team_id,
    VIDEOS_DIR
)

# Page configuration
st.set_page_config(
    page_title="Baseball Vision - Video Inference",
    page_icon="⚾",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #1565c0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">⚾ Baseball Vision - Video Inference</h1>', unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model weights selection
    weights_dir = BASE_DIR / "runs" / "baseball_detect" / "weights"
    default_weights = weights_dir / "best.pt"
    
    if default_weights.exists():
        weights_path = st.text_input(
            "Model Weights Path",
            value=str(default_weights.relative_to(BASE_DIR)),
            help="Path to the model weights file"
        )
    else:
        weights_path = st.text_input(
            "Model Weights Path",
            value="",
            help="Path to the model weights file"
        )
        st.warning("⚠️ Default weights not found. Please specify a valid path.")
    
    # Confidence threshold
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence score for detections (0.0-1.0)"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Detection Classes")
    st.markdown("""
    - **Ball** 🏀
    - **Catcher's glove** 🧤
    - **Batter's bat** 🏏
    - **Homeplate** 🏠
    """)

# Main content area - Use tabs for different video selection methods
tab1, tab2, tab3 = st.tabs(["📁 Local Videos", "📤 Upload Video", "⚾ MLB Video Search"])

selected_path = None

with tab1:
    st.header("📹 Select from Local Videos")
    
    # Get videos directory from video_downloader folder
    videos_dir = VIDEO_DOWNLOADER_DIR / "videos"
    
    if videos_dir.exists():
        video_files = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.avi")) + \
                     list(videos_dir.glob("*.mov")) + list(videos_dir.glob("*.mkv"))
        
        # Filter out uploads directory
        video_files = [f for f in video_files if "uploads" not in str(f)]
        
        if video_files:
            video_names = [f.name for f in video_files]
            selected_video = st.selectbox(
                "Choose a video",
                video_names,
                key="video_select"
            )
            selected_path = videos_dir / selected_video
            st.success(f"✅ Selected: {selected_video}")
        else:
            st.info("No videos found in the videos directory.")
    else:
        st.info("Videos directory not found.")

with tab2:
    st.header("📤 Upload a Video")
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
        key="video_upload"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        upload_dir = VIDEO_DOWNLOADER_DIR / "videos" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        temp_path = upload_dir / uploaded_file.name
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        selected_path = temp_path
        st.success(f"✅ Uploaded: {uploaded_file.name}")

with tab3:
    st.header("⚾ Search & Download MLB Videos")
    
    # Team selection
    teams = {
        "All Teams": None,
        "New York Yankees": "NYY",
        "Boston Red Sox": "BOS",
        "Tampa Bay Rays": "TB",
        "Toronto Blue Jays": "TOR",
        "Baltimore Orioles": "BAL",
        "Cleveland Guardians": "CLE",
        "Minnesota Twins": "MIN",
        "Chicago White Sox": "CWS",
        "Detroit Tigers": "DET",
        "Kansas City Royals": "KC",
        "Houston Astros": "HOU",
        "Texas Rangers": "TEX",
        "Seattle Mariners": "SEA",
        "Los Angeles Angels": "LAA",
        "Oakland Athletics": "OAK",
        "Atlanta Braves": "ATL",
        "Philadelphia Phillies": "PHI",
        "New York Mets": "NYM",
        "Miami Marlins": "MIA",
        "Washington Nationals": "WSH",
        "Milwaukee Brewers": "MIL",
        "Chicago Cubs": "CHC",
        "St. Louis Cardinals": "STL",
        "Cincinnati Reds": "CIN",
        "Pittsburgh Pirates": "PIT",
        "Los Angeles Dodgers": "LAD",
        "San Diego Padres": "SD",
        "San Francisco Giants": "SF",
        "Arizona Diamondbacks": "ARI",
        "Colorado Rockies": "COL"
    }
    
    selected_team_name = st.selectbox("Select Team", list(teams.keys()), key="team_select")
    selected_team = teams[selected_team_name]
    
    # Date selection
    col_date1, col_date2 = st.columns(2)
    with col_date1:
        search_date = st.date_input(
            "Select Date",
            value=datetime.now() - timedelta(days=1),
            max_value=datetime.now(),
            key="date_select"
        )
    
    with col_date2:
        limit = st.number_input("Max Results", min_value=1, max_value=50, value=10, key="limit_select")
    
    # Search button
    if st.button("🔍 Search MLB Videos", type="primary", use_container_width=True, key="search_mlb"):
        with st.spinner("Searching MLB Film Room..."):
            try:
                clips = search_mlb_clips(
                    date=search_date.strftime('%Y-%m-%d'),
                    team=selected_team,
                    limit=limit
                )
                
                if clips:
                    st.session_state['mlb_clips'] = clips
                    st.success(f"✅ Found {len(clips)} video(s)")
                else:
                    st.warning("⚠️ No videos found for the selected criteria.")
                    st.session_state['mlb_clips'] = []
            except Exception as e:
                st.error(f"❌ Error searching for videos: {str(e)}")
                st.session_state['mlb_clips'] = []
    
    # Display found clips
    if 'mlb_clips' in st.session_state and st.session_state['mlb_clips']:
        st.subheader("📋 Available Videos")
        
        # Store selected clips for download
        if 'selected_clip_indices' not in st.session_state:
            st.session_state['selected_clip_indices'] = []
        
        clips = st.session_state['mlb_clips']
        
        # Display clips with checkboxes
        selected_clips = []
        for i, clip in enumerate(clips):
            with st.expander(f"🎬 {clip['title']}", expanded=False):
                col_info, col_action = st.columns([3, 1])
                
                with col_info:
                    st.write(f"**Description:** {clip.get('description', 'N/A')}")
                    st.write(f"**Date:** {clip.get('date', 'N/A')}")
                    st.write(f"**Game ID:** {clip.get('game_id', 'N/A')}")
                    if clip.get('keywords'):
                        st.write(f"**Keywords:** {', '.join(clip['keywords'][:5])}")
                
                with col_action:
                    if st.checkbox("Select", key=f"clip_{i}"):
                        selected_clips.append((i, clip))
        
        # Download selected clips
        if selected_clips:
            st.markdown("---")
            st.subheader("📥 Download Selected Videos")
            
            if st.button(f"⬇️ Download {len(selected_clips)} Selected Video(s)", type="primary", use_container_width=True, key="download_clips"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                downloaded_paths = []
                download_status = st.empty()
                
                for idx, (clip_idx, clip) in enumerate(selected_clips):
                    status_text.text(f"Downloading {idx + 1}/{len(selected_clips)}: {clip['title'][:50]}...")
                    progress_bar.progress((idx) / len(selected_clips))
                    
                    # Create filename
                    safe_title = "".join(c for c in clip['title'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
                    safe_title = safe_title.replace(' ', '_')[:50]
                    filename = f"{safe_title}_{clip['game_id']}.mp4"
                    output_path = VIDEOS_DIR / filename
                    
                    try:
                        # Use a custom download function that works with Streamlit
                        response = requests.get(clip['url'], stream=True, timeout=30, headers={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                        })
                        
                        if response.status_code == 200:
                            total_size = int(response.headers.get('content-length', 0))
                            with open(output_path, 'wb') as f:
                                downloaded = 0
                                for chunk in response.iter_content(chunk_size=8192):
                                    if chunk:
                                        f.write(chunk)
                                        downloaded += len(chunk)
                                        if total_size > 0:
                                            percent = (downloaded / total_size) * 100
                                            status_text.text(f"Downloading {idx + 1}/{len(selected_clips)}: {percent:.1f}%")
                            
                            downloaded_paths.append(output_path)
                            download_status.success(f"✅ Downloaded: {filename}")
                        else:
                            download_status.error(f"❌ Failed to download: {clip['title']} (HTTP {response.status_code})")
                    except Exception as e:
                        download_status.error(f"❌ Error downloading {clip['title']}: {str(e)}")
                
                progress_bar.progress(1.0)
                status_text.text("✅ All downloads complete!")
                
                if downloaded_paths:
                    st.session_state['downloaded_paths'] = [str(p) for p in downloaded_paths]
                    st.success(f"✅ Successfully downloaded {len(downloaded_paths)} video(s)!")
                    st.info("💡 Refresh the 'Local Videos' tab to see your downloaded videos.")
                    
                    # Option to run inference immediately
                    if st.button("🚀 Run Inference on Downloaded Videos", use_container_width=True, key="inference_downloaded"):
                        st.session_state['run_inference_on'] = downloaded_paths

# Inference section (moved outside tabs)
st.markdown("---")
st.header("🎯 Run Inference")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📹 Selected Video")
    
    # Check if we should run inference on downloaded videos
    if 'run_inference_on' in st.session_state:
        selected_path = Path(st.session_state['run_inference_on'][0])
        del st.session_state['run_inference_on']
    
    if selected_path and selected_path.exists():
        st.info(f"**Selected:** {selected_path.name}")
        
        # Display video info
        file_size = selected_path.stat().st_size / (1024 * 1024)  # MB
        st.caption(f"File size: {file_size:.2f} MB")
        
        # Preview video
        try:
            st.video(str(selected_path))
        except:
            pass
    else:
        st.info("👈 Please select a video from one of the tabs above")

with col2:
    st.subheader("🚀 Inference Controls")
    
    if selected_path and selected_path.exists():
        # Run inference button
        if st.button("🚀 Run Inference", type="primary", use_container_width=True):
            if not weights_path or not Path(weights_path).exists():
                if not default_weights.exists():
                    st.error("❌ Model weights not found. Please specify a valid path in settings.")
                else:
                    weights_path = str(default_weights)
            
            # Validate weights path
            weights_full_path = Path(weights_path)
            if not weights_full_path.is_absolute():
                weights_full_path = BASE_DIR / weights_path
            
            if not weights_full_path.exists():
                st.error(f"❌ Weights file not found: {weights_full_path}")
            else:
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🔄 Loading model...")
                progress_bar.progress(10)
                
                try:
                    status_text.text("🔍 Running inference on video...")
                    progress_bar.progress(30)
                    
                    # Run inference
                    results = predict(
                        source=str(selected_path),
                        weights_path=str(weights_full_path),
                        conf=conf_threshold
                    )
                    
                    progress_bar.progress(90)
                    status_text.text("💾 Saving results...")
                    
                    if results:
                        progress_bar.progress(100)
                        status_text.text("✅ Inference complete!")
                        
                        # Get output path
                        output_path = BASE_DIR / "runs" / "predictions" / selected_path.name
                        
                        st.success("✅ Inference completed successfully!")
                        st.markdown(f"**Output saved to:** `{output_path.relative_to(BASE_DIR)}`")
                        
                        # Show output video if it exists
                        if output_path.exists():
                            st.video(str(output_path))
                            
                            # Download button
                            with open(output_path, "rb") as video_file:
                                st.download_button(
                                    label="📥 Download Result Video",
                                    data=video_file,
                                    file_name=output_path.name,
                                    mime="video/mp4"
                                )
                        else:
                            st.warning("⚠️ Output video not found. Check the runs/predictions directory.")
                    else:
                        st.error("❌ Inference failed. Check the console for error messages.")
                        
                except Exception as e:
                    st.error(f"❌ Error during inference: {str(e)}")
                    st.exception(e)
                finally:
                    progress_bar.empty()
                    status_text.empty()
    else:
        st.info("👈 Please select or upload a video to run inference")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Baseball Vision - YOLO Object Detection for Baseball Videos</p>
    <p>Detecting: Ball, Catcher's glove, Batter's bat, Homeplate</p>
</div>
""", unsafe_allow_html=True)

