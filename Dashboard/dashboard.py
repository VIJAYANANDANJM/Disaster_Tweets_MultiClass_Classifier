"""
Main Dashboard Application
Desktop app for tweet classification with XAI visualization.
Gets tweets from database and allows manual input.
"""
import customtkinter as ctk
from tkinter import ttk
import threading
from datetime import datetime
from typing import List, Dict, Optional
import uuid

from Dashboard.api_client import APIClient
from Dashboard.model_inference import ModelInference
from Dashboard.token_highlighter import TokenHighlighter
from Dashboard.config import (
    LABEL_DISPLAY_NAMES, LABEL_COLORS, MAX_TWEETS_PER_FETCH
)

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ConnectionFrame(ctk.CTkFrame):
    """Backend connection check frame."""
    
    def __init__(self, parent, on_success_callback):
        super().__init__(parent)
        self.on_success = on_success_callback
        self.api_client = APIClient()
        
        self.setup_ui()
        
        # Check if backend is running
        self.check_backend_status()
    
    def setup_ui(self):
        """Setup connection UI components."""
        # Title
        title = ctk.CTkLabel(
            self,
            text="Disaster Tweet Classification Dashboard",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(pady=30)
        
        # Subtitle
        subtitle = ctk.CTkLabel(
            self,
            text="Connecting to backend server...",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        subtitle.pack(pady=10)
        
        # Status label
        self.status_label = ctk.CTkLabel(
            self,
            text="Checking connection...",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(pady=10)
        
        # Retry button
        self.retry_btn = ctk.CTkButton(
            self,
            text="Retry Connection",
            command=self.check_backend_status,
            width=200,
            height=40,
            font=ctk.CTkFont(size=14)
        )
        self.retry_btn.pack(pady=20)
        self.retry_btn.pack_forget()  # Hide initially
    
    def check_backend_status(self):
        """Check if backend server is running."""
        def check_thread():
            if self.api_client.check_health():
                self.status_label.configure(
                    text="✓ Backend server connected",
                    text_color="green"
                )
                self.retry_btn.pack_forget()
                self.after(500, self.on_success)
            else:
                self.status_label.configure(
                    text="✗ Backend server not running. Please start the backend server first.",
                    text_color="red"
                )
                self.retry_btn.pack()
        
        threading.Thread(target=check_thread, daemon=True).start()


class TweetInputFrame(ctk.CTkFrame):
    """Frame for manual tweet input."""
    
    def __init__(self, parent, on_submit_callback):
        super().__init__(parent)
        self.on_submit = on_submit_callback
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup tweet input UI."""
        title = ctk.CTkLabel(
            self,
            text="Enter Tweet",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title.pack(pady=10)
        
        # Text area for tweet input
        self.tweet_text = ctk.CTkTextbox(
            self,
            width=500,
            height=100,
            font=ctk.CTkFont(size=12)
        )
        self.tweet_text.pack(pady=10, padx=10, fill="x")
        
        # Author input
        author_frame = ctk.CTkFrame(self, fg_color="transparent")
        author_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(
            author_frame,
            text="Author:",
            font=ctk.CTkFont(size=12)
        ).pack(side="left", padx=5)
        
        self.author_entry = ctk.CTkEntry(
            author_frame,
            placeholder_text="Enter author username (optional)",
            width=300
        )
        self.author_entry.pack(side="left", padx=5, fill="x", expand=True)
        
        # Submit button
        submit_btn = ctk.CTkButton(
            self,
            text="Classify Tweet",
            command=self.submit_tweet,
            width=200,
            height=35,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        submit_btn.pack(pady=10)
        
        # Status label
        self.status_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=11)
        )
        self.status_label.pack(pady=5)
    
    def submit_tweet(self):
        """Handle tweet submission."""
        text = self.tweet_text.get("1.0", "end-1c").strip()
        author = self.author_entry.get().strip() or "manual_input"
        
        if not text:
            self.status_label.configure(
                text="Please enter tweet text",
                text_color="red"
            )
            return
        
        if len(text) < 5:
            self.status_label.configure(
                text="Tweet text is too short",
                text_color="red"
            )
            return
        
        self.status_label.configure(
            text="Processing...",
            text_color="yellow"
        )
        
        # Create tweet object
        tweet_data = {
            'text': text,
            'author': author,
            'authorName': author,
            'createdAt': datetime.now().isoformat(),
            'tweetId': str(uuid.uuid4()),
            'retweetCount': 0,
            'favoriteCount': 0
        }
        
        # Call callback
        self.on_submit(tweet_data)
        
        # Clear input
        self.tweet_text.delete("1.0", "end")
        self.author_entry.delete(0, "end")
        self.status_label.configure(
            text="Tweet submitted successfully!",
            text_color="green"
        )


class TweetCard(ctk.CTkFrame):
    """Individual tweet card component."""
    
    def __init__(self, parent, tweet_data, classification_result, on_select_callback):
        super().__init__(parent, corner_radius=10)
        self.tweet_data = tweet_data
        self.classification = classification_result
        self.on_select = on_select_callback
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup tweet card UI."""
        # Label badge
        label_id = self.classification.get("predictedLabelId")
        if label_id is not None:
            label_name = LABEL_DISPLAY_NAMES.get(label_id, "Unknown")
            label_color = LABEL_COLORS.get(label_id, "#95A5A6")
            
            label_frame = ctk.CTkFrame(self, fg_color=label_color, corner_radius=5)
            label_frame.pack(fill="x", padx=10, pady=(10, 5))
            
            label_text = ctk.CTkLabel(
                label_frame,
                text=label_name,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="white"
            )
            label_text.pack(pady=5)
        else:
            # Not classified yet
            label_frame = ctk.CTkFrame(self, fg_color="#95A5A6", corner_radius=5)
            label_frame.pack(fill="x", padx=10, pady=(10, 5))
            
            label_text = ctk.CTkLabel(
                label_frame,
                text="Not Classified",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="white"
            )
            label_text.pack(pady=5)
        
        # Author and time
        info_frame = ctk.CTkFrame(self, fg_color="transparent")
        info_frame.pack(fill="x", padx=10, pady=5)
        
        author_label = ctk.CTkLabel(
            info_frame,
            text=f"@{self.tweet_data.get('author', 'unknown')}",
            font=ctk.CTkFont(size=11, weight="bold")
        )
        author_label.pack(side="left")
        
        created_at = self.tweet_data.get('createdAt')
        if isinstance(created_at, str):
            time_str = created_at[:16] if len(created_at) > 16 else created_at
        elif hasattr(created_at, 'strftime'):
            time_str = created_at.strftime("%Y-%m-%d %H:%M")
        else:
            time_str = str(created_at)[:16] if created_at else "Unknown"
        
        time_label = ctk.CTkLabel(
            info_frame,
            text=time_str,
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        time_label.pack(side="right")
        
        # Tweet text (truncated)
        tweet_text = self.tweet_data.get('text', '')
        if len(tweet_text) > 150:
            tweet_text = tweet_text[:150] + "..."
        
        text_label = ctk.CTkLabel(
            self,
            text=tweet_text,
            font=ctk.CTkFont(size=12),
            wraplength=500,
            justify="left",
            anchor="w"
        )
        text_label.pack(fill="x", padx=10, pady=5)
        
        # Confidence score
        confidence_scores = self.classification.get("confidenceScores", [])
        if confidence_scores:
            confidence = max(confidence_scores)
            conf_label = ctk.CTkLabel(
                self,
                text=f"Confidence: {confidence:.2%}",
                font=ctk.CTkFont(size=10),
                text_color="gray"
            )
            conf_label.pack(pady=5)
        
        # Click to select
        self.bind("<Button-1>", lambda e: self.on_select())
        for widget in self.winfo_children():
            widget.bind("<Button-1>", lambda e: self.on_select())
        
        self.configure(cursor="hand2")


class TweetDetailFrame(ctk.CTkFrame):
    """Detailed tweet view with actionable info and token highlighting."""
    
    def __init__(self, parent, token_highlighter):
        super().__init__(parent)
        self.token_highlighter = token_highlighter
        self.current_tweet = None
        self.current_classification = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup detail view UI."""
        # Title
        title = ctk.CTkLabel(
            self,
            text="Tweet Details",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title.pack(pady=10)
        
        # Scrollable frame for content
        scroll_frame = ctk.CTkScrollableFrame(self)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Tweet info section
        self.info_frame = ctk.CTkFrame(scroll_frame)
        self.info_frame.pack(fill="x", pady=5)
        
        # Highlighted tweet text
        self.text_frame = ctk.CTkFrame(scroll_frame)
        self.text_frame.pack(fill="x", pady=5)
        
        # Actionable info section
        self.actionable_frame = ctk.CTkFrame(scroll_frame)
        self.actionable_frame.pack(fill="x", pady=5)
        
        # Initially show placeholder
        placeholder = ctk.CTkLabel(
            scroll_frame,
            text="Select a tweet to view details",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        placeholder.pack(pady=50)
        self.placeholder = placeholder
    
    def display_tweet(self, tweet_data):
        """Display tweet details."""
        self.current_tweet = tweet_data
        
        # Extract classification from tweet data
        classification = tweet_data.get('classification', {})
        self.current_classification = classification
        
        # Hide placeholder
        if hasattr(self, 'placeholder'):
            self.placeholder.pack_forget()
        
        # Clear previous content
        for widget in self.info_frame.winfo_children():
            widget.destroy()
        for widget in self.text_frame.winfo_children():
            widget.destroy()
        for widget in self.actionable_frame.winfo_children():
            widget.destroy()
        
        # Tweet info
        label_id = classification.get("predictedLabelId")
        if label_id is not None:
            label_name = LABEL_DISPLAY_NAMES.get(label_id, "Unknown")
            label_color = LABEL_COLORS.get(label_id, "#95A5A6")
        else:
            label_name = "Not Classified"
            label_color = "#95A5A6"
        
        info_title = ctk.CTkLabel(
            self.info_frame,
            text="Classification",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        info_title.pack(anchor="w", padx=10, pady=5)
        
        label_badge = ctk.CTkFrame(self.info_frame, fg_color=label_color, corner_radius=5)
        label_badge.pack(fill="x", padx=10, pady=5)
        
        label_text = ctk.CTkLabel(
            label_badge,
            text=label_name,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="white"
        )
        label_text.pack(pady=8)
        
        confidence_scores = classification.get("confidenceScores", [])
        if confidence_scores:
            confidence = max(confidence_scores)
            conf_text = ctk.CTkLabel(
                self.info_frame,
                text=f"Confidence: {confidence:.2%}",
                font=ctk.CTkFont(size=12)
            )
            conf_text.pack(anchor="w", padx=10, pady=5)
        
        author_text = ctk.CTkLabel(
            self.info_frame,
            text=f"Author: @{tweet_data.get('author', 'unknown')} ({tweet_data.get('authorName', 'Unknown')})",
            font=ctk.CTkFont(size=12)
        )
        author_text.pack(anchor="w", padx=10, pady=5)
        
        created_at = tweet_data.get('createdAt')
        if isinstance(created_at, str):
            time_str = created_at
        elif hasattr(created_at, 'strftime'):
            time_str = created_at.strftime('%Y-%m-%d %H:%M:%S')
        else:
            time_str = str(created_at)
        
        time_text = ctk.CTkLabel(
            self.info_frame,
            text=f"Time: {time_str}",
            font=ctk.CTkFont(size=12)
        )
        time_text.pack(anchor="w", padx=10, pady=5)
        
        # Highlighted tweet text
        text_title = ctk.CTkLabel(
            self.text_frame,
            text="Tweet Text (Token Importance Highlighted)",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        text_title.pack(anchor="w", padx=10, pady=5)
        
        # Create highlighted text
        explanation = tweet_data.get("explanation", [])
        if explanation:
            # Convert explanation format if needed
            if isinstance(explanation, list) and len(explanation) > 0:
                if isinstance(explanation[0], dict):
                    # Convert from dict format to tuple format
                    explanation = [(e.get('token', ''), e.get('score', 0.0)) for e in explanation]
            
            segments = self.token_highlighter.create_highlighted_text(
                tweet_data.get('text', ''),
                explanation
            )
            
            # Display segments with colors
            text_container = ctk.CTkFrame(self.text_frame)
            text_container.pack(fill="x", padx=10, pady=10)
            
            for segment_text, color, score in segments:
                if segment_text.strip():
                    label = ctk.CTkLabel(
                        text_container,
                        text=segment_text,
                        font=ctk.CTkFont(size=13),
                        fg_color=color if score > 0 else "transparent",
                        text_color="white" if score > 0.5 else "black",
                        corner_radius=3,
                        padx=2
                    )
                    label.pack(side="left")
        else:
            plain_text = ctk.CTkLabel(
                self.text_frame,
                text=tweet_data.get('text', ''),
                font=ctk.CTkFont(size=13),
                wraplength=600,
                justify="left"
            )
            plain_text.pack(anchor="w", padx=10, pady=10)
        
        # Actionable info
        actionable_info = tweet_data.get("actionableInfo")
        if actionable_info:
            actionable_title = ctk.CTkLabel(
                self.actionable_frame,
                text="Actionable Information",
                font=ctk.CTkFont(size=16, weight="bold")
            )
            actionable_title.pack(anchor="w", padx=10, pady=5)
            
            # Locations
            if actionable_info.get("locations"):
                loc_text = ctk.CTkLabel(
                    self.actionable_frame,
                    text=f"📍 Locations: {', '.join(actionable_info['locations'])}",
                    font=ctk.CTkFont(size=12)
                )
                loc_text.pack(anchor="w", padx=10, pady=5)
            
            # People count
            if actionable_info.get("people_count"):
                people_texts = [
                    f"{p['count']} {p['status']}" 
                    for p in actionable_info['people_count']
                ]
                people_text = ctk.CTkLabel(
                    self.actionable_frame,
                    text=f"👥 People: {', '.join(people_texts)}",
                    font=ctk.CTkFont(size=12)
                )
                people_text.pack(anchor="w", padx=10, pady=5)
            
            # Needs
            if actionable_info.get("needs"):
                needs_text = ctk.CTkLabel(
                    self.actionable_frame,
                    text=f"🆘 Needs: {', '.join(actionable_info['needs'])}",
                    font=ctk.CTkFont(size=12)
                )
                needs_text.pack(anchor="w", padx=10, pady=5)
            
            # Damage type
            if actionable_info.get("damage_type"):
                damage_text = ctk.CTkLabel(
                    self.actionable_frame,
                    text=f"💥 Damage: {', '.join(actionable_info['damage_type'])}",
                    font=ctk.CTkFont(size=12)
                )
                damage_text.pack(anchor="w", padx=10, pady=5)
            
            # Time mentions
            if actionable_info.get("time_mentions"):
                time_text = ctk.CTkLabel(
                    self.actionable_frame,
                    text=f"⏰ Time: {', '.join(actionable_info['time_mentions'])}",
                    font=ctk.CTkFont(size=12)
                )
                time_text.pack(anchor="w", padx=10, pady=5)
        else:
            no_actionable = ctk.CTkLabel(
                self.actionable_frame,
                text="No actionable information extracted for this tweet.",
                font=ctk.CTkFont(size=12),
                text_color="gray"
            )
            no_actionable.pack(anchor="w", padx=10, pady=10)


class MainDashboard(ctk.CTk):
    """Main dashboard application."""
    
    def __init__(self):
        super().__init__()
        
        self.title("Disaster Tweet Classification Dashboard")
        self.geometry("1400x900")
        
        self.api_client = APIClient()
        self.token_highlighter = TokenHighlighter(
            min_color="#FFFFFF",
            max_color="#FF6B6B"  # Red for high importance
        )
        self.model_inference = None  # Will be initialized when needed
        
        self.tweets = []
        self.current_filter = None
        
        self.setup_ui()
        
        # Show connection screen first
        self.show_connection()
    
    def setup_ui(self):
        """Setup main UI structure."""
        # Main container
        self.main_container = ctk.CTkFrame(self)
        self.main_container.pack(fill="both", expand=True)
        
        # Connection frame (shown first)
        self.connection_frame = ConnectionFrame(self.main_container, self.on_connection_success)
        
        # Dashboard frame (shown after connection)
        self.dashboard_frame = ctk.CTkFrame(self.main_container)
        self.setup_dashboard()
    
    def show_connection(self):
        """Show connection screen."""
        self.dashboard_frame.pack_forget()
        self.connection_frame.pack(fill="both", expand=True)
    
    def on_connection_success(self):
        """Handle successful connection."""
        self.connection_frame.pack_forget()
        self.dashboard_frame.pack(fill="both", expand=True)
        self.load_tweets_from_db()
    
    def setup_dashboard(self):
        """Setup dashboard UI."""
        # Top bar
        top_bar = ctk.CTkFrame(self.dashboard_frame)
        top_bar.pack(fill="x", padx=10, pady=10)
        
        title = ctk.CTkLabel(
            top_bar,
            text="Disaster Tweet Classifier",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(side="left", padx=10)
        
        # Connection status
        self.connection_status_label = ctk.CTkLabel(
            top_bar,
            text="✓ Connected",
            font=ctk.CTkFont(size=12),
            text_color="green"
        )
        self.connection_status_label.pack(side="right", padx=10)
        
        # Main content area (split view)
        content_frame = ctk.CTkFrame(self.dashboard_frame)
        content_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left panel - Tweet input and list
        left_panel = ctk.CTkFrame(content_frame)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        # Tweet input frame
        self.tweet_input_frame = TweetInputFrame(left_panel, self.handle_manual_tweet)
        self.tweet_input_frame.pack(fill="x", padx=10, pady=10)
        
        # Filter section
        filter_frame = ctk.CTkFrame(left_panel)
        filter_frame.pack(fill="x", padx=10, pady=10)
        
        filter_label = ctk.CTkLabel(
            filter_frame,
            text="Filter by Category:",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        filter_label.pack(side="left", padx=10)
        
        filter_buttons_frame = ctk.CTkFrame(filter_frame, fg_color="transparent")
        filter_buttons_frame.pack(side="left", padx=10)
        
        # All button
        all_btn = ctk.CTkButton(
            filter_buttons_frame,
            text="All",
            command=lambda: self.filter_tweets(None),
            width=100,
            height=30
        )
        all_btn.pack(side="left", padx=5)
        
        # Category filter buttons
        for label_id, label_name in LABEL_DISPLAY_NAMES.items():
            btn = ctk.CTkButton(
                filter_buttons_frame,
                text=label_name,
                command=lambda lid=label_id: self.filter_tweets(lid),
                width=150,
                height=30,
                fg_color=LABEL_COLORS.get(label_id, "#95A5A6")
            )
            btn.pack(side="left", padx=5)
        
        # Refresh button
        refresh_btn = ctk.CTkButton(
            filter_frame,
            text="Refresh",
            command=self.load_tweets_from_db,
            width=100,
            height=30,
            font=ctk.CTkFont(size=12)
        )
        refresh_btn.pack(side="right", padx=5)
        
        # Tweet list scrollable frame
        self.tweet_list_frame = ctk.CTkScrollableFrame(left_panel)
        self.tweet_list_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Right panel - Tweet detail
        right_panel = ctk.CTkFrame(content_frame)
        right_panel.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        self.detail_frame = TweetDetailFrame(
            right_panel,
            self.token_highlighter
        )
        self.detail_frame.pack(fill="both", expand=True)
    
    def handle_manual_tweet(self, tweet_data):
        """Handle manually entered tweet."""
        def process_thread():
            # Classify the tweet locally
            self.classify_single_tweet(tweet_data)
            # Save to database
            self.save_tweet_to_db(tweet_data)
            # Refresh list
            self.load_tweets_from_db()
        
        threading.Thread(target=process_thread, daemon=True).start()
    
    def classify_single_tweet(self, tweet_data):
        """Classify a single tweet locally."""
        if not self.model_inference:
            from Dashboard.model_inference import ModelInference
            self.model_inference = ModelInference()
        
        if not self.model_inference.loaded:
            return
        
        text = tweet_data.get('text', '')
        if text:
            result = self.model_inference.classify_tweet(text)
            if result:
                tweet_data['classification'] = {
                    'predictedLabelId': result['predicted_label_id'],
                    'predictedLabel': result['predicted_label'],
                    'confidenceScores': result['confidence_scores'],
                    'classifiedAt': datetime.now().isoformat()
                }
                tweet_data['explanation'] = result.get('explanation', [])
                tweet_data['actionableInfo'] = result.get('actionable_info', {})
    
    def save_tweet_to_db(self, tweet_data):
        """Save tweet to database via backend."""
        try:
            success, message, saved_tweet = self.api_client.create_tweet(tweet_data)
            if success:
                print(f"Tweet saved to database: {message}")
            else:
                print(f"Error saving tweet: {message}")
        except Exception as e:
            print(f"Error saving tweet: {e}")
    
    def load_tweets_from_db(self):
        """Load tweets from database and classify unclassified ones locally."""
        def load_thread():
            label_id = self.current_filter if self.current_filter is not None else None
            success, message, tweets, pagination = self.api_client.get_tweets(
                page=1,
                limit=MAX_TWEETS_PER_FETCH,
                label_id=label_id
            )
            
            if success:
                # Classify unclassified tweets locally
                self.classify_tweets_locally(tweets)
            else:
                print(f"Error: {message}")
        
        threading.Thread(target=load_thread, daemon=True).start()
    
    def classify_tweets_locally(self, tweets):
        """Classify tweets locally using the model."""
        if not self.model_inference:
            from Dashboard.model_inference import ModelInference
            self.model_inference = ModelInference()
        
        if not self.model_inference.loaded:
            print("Error: Model not loaded")
            self.tweets = tweets
            self.display_tweets()
            return
        
        # Classify each unclassified tweet
        classified_tweets = []
        for tweet in tweets:
            # Check if already classified
            if tweet.get('classification') and tweet.get('classification').get('predictedLabelId') is not None:
                classified_tweets.append(tweet)
                continue
            
            # Classify locally
            text = tweet.get('text', '')
            if text:
                result = self.model_inference.classify_tweet(text)
                if result:
                    # Update tweet with classification
                    tweet['classification'] = {
                        'predictedLabelId': result['predicted_label_id'],
                        'predictedLabel': result['predicted_label'],
                        'confidenceScores': result['confidence_scores'],
                        'classifiedAt': datetime.now().isoformat()
                    }
                    tweet['explanation'] = result.get('explanation', [])
                    tweet['actionableInfo'] = result.get('actionable_info', {})
                    
                    # Save classification to backend (async, non-blocking)
                    self.save_classification_to_backend(tweet)
                
                classified_tweets.append(tweet)
            else:
                classified_tweets.append(tweet)
        
        self.tweets = classified_tweets
        self.display_tweets()
    
    def save_classification_to_backend(self, tweet):
        """Save classification results to backend (async, non-blocking)."""
        def save_thread():
            try:
                tweet_id = tweet.get('tweetId')
                if tweet_id:
                    self.api_client.update_tweet_classification(
                        tweet_id,
                        tweet.get('classification'),
                        tweet.get('explanation', []),
                        tweet.get('actionableInfo', {})
                    )
            except Exception as e:
                print(f"Error saving classification: {e}")
        
        threading.Thread(target=save_thread, daemon=True).start()
    
    def display_tweets(self):
        """Display tweets in the list."""
        # Clear existing tweets
        for widget in self.tweet_list_frame.winfo_children():
            widget.destroy()
        
        # Filter tweets if filter is active (already filtered from API)
        display_tweets = self.tweets
        if self.current_filter is not None:
            display_tweets = [
                t for t in self.tweets
                if t.get('classification', {}).get('predictedLabelId') == self.current_filter
            ]
        
        # Create tweet cards
        for tweet in display_tweets:
            classification = tweet.get('classification', {})
            card = TweetCard(
                self.tweet_list_frame,
                tweet,
                classification,
                lambda t=tweet: self.select_tweet(t)
            )
            card.pack(fill="x", padx=5, pady=5)
    
    def filter_tweets(self, label_id):
        """Filter tweets by label."""
        self.current_filter = label_id
        self.load_tweets_from_db()
    
    def select_tweet(self, tweet):
        """Handle tweet selection."""
        self.detail_frame.display_tweet(tweet)


def main():
    """Main entry point."""
    app = MainDashboard()
    app.mainloop()


if __name__ == "__main__":
    main()
