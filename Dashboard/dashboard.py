"""
Main Dashboard Application
Desktop app for tweet classification with XAI visualization.
Gets tweets from database and allows manual input.
Includes HITL (Human-In-The-Loop) Database Synchronization and Refresh logic.
"""
import customtkinter as ctk
from tkinter import ttk
import threading
from datetime import datetime
from typing import List, Dict, Optional, Any
import uuid

from Dashboard.api_client import APIClient, _normalize_actionable_info
from Dashboard.model_inference import ModelInference
from Dashboard.token_highlighter import TokenHighlighter
from Dashboard.config import (
    LABEL_DISPLAY_NAMES, LABEL_COLORS, MAX_TWEETS_PER_FETCH, ACTIONABLE_LABELS
)

# HITL Storage
from Dashboard.HITL.feedback_storage import save_feedback

# Configuration Constants
CONFIDENCE_THRESHOLD = 0.7

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
                    text="Backend server connected",
                    text_color="green"
                )
                self.retry_btn.pack_forget()
                self.after(500, self.on_success)
            else:
                self.status_label.configure(
                    text="Backend server not running. Please start the backend server first.",
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
        ) .pack(side="left", padx=5)
        
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
        
        # Create tweet object matching the new status schema
        tweet_data = {
            'text': text,
            'author': author,
            'authorName': author,
            'authorId': author,
            'createdAt': datetime.now().isoformat(),
            'tweetId': str(uuid.uuid4()),
            'retweetCount': 0,
            'favoriteCount': 0,
            'status': 'unverified'  # Default state before threshold check
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
        status = self.tweet_data.get('status', 'unverified')

        if label_id is not None:
            label_name = LABEL_DISPLAY_NAMES.get(label_id, "Unknown")
            
            # Show specific badge text based on status
            if status == 'human_verified':
                label_name += " (Human Verified)"
            elif status == 'verified':
                label_name += " (Auto Verified)"
            else:
                label_name += " (To Verify)"
                
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
            wraplength=450,
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

    def __init__(self, parent, token_highlighter, api_client, dashboard_ref):
        super().__init__(parent)
        self.token_highlighter = token_highlighter
        self.api_client = api_client  # For DB syncing
        self.dashboard = dashboard_ref  # Reference to trigger global refresh
        self.current_tweet = None
        self.current_classification = None
        self.verification_mode = False

        self.setup_ui()

    def setup_ui(self):
        """Setup detail view UI."""
        header_container = ctk.CTkFrame(self, fg_color="transparent")
        header_container.pack(fill="x", pady=10)

        title = ctk.CTkLabel(
            header_container,
            text="Tweet Details",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title.pack(side="left", padx=20)

        self.scroll_frame = ctk.CTkScrollableFrame(self)
        self.scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.info_frame = ctk.CTkFrame(self.scroll_frame)
        self.info_frame.pack(fill="x", pady=5)

        self.text_frame = ctk.CTkFrame(self.scroll_frame)
        self.text_frame.pack(fill="x", pady=5)

        self.actionable_frame = ctk.CTkFrame(self.scroll_frame)
        self.actionable_frame.pack(fill="x", pady=5)

        self.placeholder = ctk.CTkLabel(
            self.scroll_frame,
            text="Select a tweet to view details",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        self.placeholder.pack(pady=50)

    def sync_to_db(self, tweet_data):
        """Update the database via the API client when a human makes a choice."""
        def run_sync():
            try:
                self.api_client.update_tweet_classification(
                    tweet_data['tweetId'],
                    tweet_data['classification'],
                    tweet_data.get('explanation', []),
                    tweet_data.get('actionableInfo', {}),
                    status=tweet_data.get('status', 'verified')
                )
            except Exception as e:
                print(f"Error syncing to DB: {e}")

        threading.Thread(target=run_sync, daemon=True).start()

    def open_correction_popup(self, tweet_data, current_label_id):
        """Centered popup for selecting correct class."""
        popup = ctk.CTkToplevel(self)
        popup.title("Human Classification Correction")
        popup.geometry("600x550")

        popup.update_idletasks()
        x = (popup.winfo_screenwidth() // 2) - (600 // 2)
        y = (popup.winfo_screenheight() // 2) - (550 // 2)
        popup.geometry(f"+{x}+{y}")
        popup.attributes("-topmost", True)

        ctk.CTkLabel(
            popup,
            text="Verify and Correct Classification",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=15)

        text_box = ctk.CTkTextbox(popup, height=120, width=500)
        text_box.insert("1.0", f"TWEET CONTENT:\n\n{tweet_data.get('text', '')}")
        text_box.configure(state="disabled")
        text_box.pack(pady=10, padx=20)

        ctk.CTkLabel(
            popup,
            text="Please select the correct category:",
            font=ctk.CTkFont(weight="bold")
        ).pack(pady=10)

        choice_var = ctk.IntVar(value=current_label_id if current_label_id is not None else 0)

        for lid, name in LABEL_DISPLAY_NAMES.items():
            ctk.CTkRadioButton(
                popup,
                text=name,
                variable=choice_var,
                value=lid
            ).pack(pady=4, anchor="w", padx=180)

        def submit_correction():
            selected_lid = choice_var.get()
            tweet_data['status'] = 'human_verified'
            if 'classification' in tweet_data:
                tweet_data['classification']['predictedLabelId'] = selected_lid
                tweet_data['classification']['predictedLabel'] = LABEL_DISPLAY_NAMES.get(selected_lid, "Unknown")

            self.sync_to_db(tweet_data)
            save_feedback(tweet_data.get('text', ''), selected_lid)
            popup.destroy()
            self.display_tweet(tweet_data)
            self.dashboard.display_tweets()

        ctk.CTkButton(
            popup,
            text="Submit Correction & Sync to DB",
            command=submit_correction,
            fg_color="#1F6AA5",
            height=40
        ).pack(pady=25)

    def display_tweet(self, tweet_data):
        """Display tweet details."""
        self.current_tweet = tweet_data
        classification = tweet_data.get('classification', {})
        self.current_classification = classification

        if hasattr(self, 'placeholder'):
            self.placeholder.pack_forget()

        for widget in self.info_frame.winfo_children():
            widget.destroy()
        for widget in self.text_frame.winfo_children():
            widget.destroy()
        for widget in self.actionable_frame.winfo_children():
            widget.destroy()

        label_id = classification.get("predictedLabelId")
        label_name = LABEL_DISPLAY_NAMES.get(label_id, "Unknown") if label_id is not None else "Not Classified"
        label_color = LABEL_COLORS.get(label_id, "#95A5A6")

        current_status = tweet_data.get('status', 'unverified')
        status_text = "Model Classification"
        if current_status == 'human_verified':
            status_text += " (Human Verified & Stored)"
        elif current_status == 'verified':
            status_text += " (Auto Verified)"
        else:
            status_text += " (Requires Human Review)"

        info_title = ctk.CTkLabel(self.info_frame, text=status_text, font=ctk.CTkFont(size=16, weight="bold"))
        if current_status != 'unverified':
            info_title.configure(text_color="#27AE60")
        else:
            info_title.configure(text_color="#E67E22")
        info_title.pack(anchor="w", padx=10, pady=5)

        label_badge = ctk.CTkFrame(self.info_frame, fg_color=label_color, corner_radius=5)
        label_badge.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(
            label_badge,
            text=label_name,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="white"
        ).pack(pady=8)

        if self.verification_mode and current_status == 'unverified':
            verify_box = ctk.CTkFrame(self.info_frame, fg_color="#333333", border_width=1)
            verify_box.pack(fill="x", padx=10, pady=10)

            ctk.CTkLabel(
                verify_box,
                text="Is this classification correct?",
                font=ctk.CTkFont(weight="bold")
            ).pack(pady=5)

            btn_row = ctk.CTkFrame(verify_box, fg_color="transparent")
            btn_row.pack(pady=5)

            def on_accept():
                tweet_data['status'] = 'human_verified'
                self.sync_to_db(tweet_data)
                save_feedback(tweet_data.get('text', ''), label_id)
                accept_btn.configure(text="Synced OK", state="disabled")
                reject_btn.pack_forget()
                info_title.configure(text="Model Classification (Human Verified & Stored)", text_color="#27AE60")
                self.dashboard.display_tweets()

            accept_btn = ctk.CTkButton(btn_row, text="Accept", fg_color="#27AE60", width=120, command=on_accept)
            accept_btn.pack(side="left", padx=10)

            reject_btn = ctk.CTkButton(
                btn_row,
                text="Reject & Correct",
                fg_color="#C0392B",
                width=120,
                command=lambda: self.open_correction_popup(tweet_data, label_id)
            )
            reject_btn.pack(side="left", padx=10)

        elif current_status == 'human_verified' or current_status == 'verified':
            sync_msg = ctk.CTkLabel(
                self.info_frame,
                text="Classification Sync Complete",
                text_color="#27AE60",
                font=ctk.CTkFont(size=11)
            )
            sync_msg.pack(pady=5)

        meta_frame = ctk.CTkFrame(self.info_frame, fg_color="transparent")
        meta_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(
            meta_frame,
            text=f"Author: @{tweet_data.get('author', 'unknown')}",
            font=ctk.CTkFont(size=12, weight="bold")
        ).pack(anchor="w")

        text_title = ctk.CTkLabel(self.text_frame, text="Token Importance (XAI)", font=ctk.CTkFont(size=16, weight="bold"))
        text_title.pack(anchor="w", padx=10, pady=5)

        explanation = tweet_data.get("explanation", [])
        if explanation:
            if isinstance(explanation, list) and len(explanation) > 0 and isinstance(explanation[0], dict):
                explanation = [(e.get('token', ''), e.get('score', 0.0)) for e in explanation]

            segments = self.token_highlighter.create_highlighted_text(tweet_data.get('text', ''), explanation)
            text_container = ctk.CTkFrame(self.text_frame, fg_color="#1A1A1A")
            text_container.pack(fill="x", padx=10, pady=10)
            inner_wrap = ctk.CTkFrame(text_container, fg_color="transparent")
            inner_wrap.pack(padx=5, pady=5)

            for segment_text, color, score in segments:
                if segment_text.strip() or " " in segment_text:
                    lbl = ctk.CTkLabel(
                        inner_wrap,
                        text=segment_text,
                        fg_color=color if score > 0.1 else "transparent",
                        text_color="white" if score > 0.4 else "gray90",
                        corner_radius=2,
                        padx=1
                    )
                    lbl.pack(side="left")
        else:
            ctk.CTkLabel(
                self.text_frame,
                text=tweet_data.get('text', ''),
                wraplength=550,
                justify="left"
            ).pack(padx=10, pady=10)

        actionable_info = tweet_data.get("actionableInfo")
        if isinstance(actionable_info, dict) and actionable_info:
            def _to_str_list(value):
                if value is None:
                    return []
                if not isinstance(value, (list, tuple)):
                    value = [value]
                return [str(v).strip() for v in value if str(v).strip()]

            locations = _to_str_list(actionable_info.get("locations"))
            needs = _to_str_list(actionable_info.get("needs"))
            damage = _to_str_list(actionable_info.get("damageType") or actionable_info.get("damage_type"))
            time_mentions = _to_str_list(actionable_info.get("timeMentions") or actionable_info.get("time_mentions"))

            people_raw = actionable_info.get("peopleCount") or actionable_info.get("people_count") or []
            if not isinstance(people_raw, (list, tuple)):
                people_raw = [people_raw]

            people_count = []
            for item in people_raw:
                if isinstance(item, dict):
                    count = item.get("count")
                    status = item.get("status")
                    if count is not None and status:
                        people_count.append(f"{count} {status}")
                    elif count is not None:
                        people_count.append(str(count))
                    elif status:
                        people_count.append(str(status))
                elif str(item).strip():
                    people_count.append(str(item).strip())

            location_note = actionable_info.get("locationNote") or actionable_info.get("location_note")
            has_data = any([locations, needs, damage, people_count, time_mentions, location_note])

            if has_data:
                ctk.CTkLabel(
                    self.actionable_frame,
                    text="Actionable Extraction",
                    font=ctk.CTkFont(size=16, weight="bold")
                ).pack(anchor="w", padx=10, pady=5)

                rows = [
                    ("Locations", locations),
                    ("Needs", needs),
                    ("Damage Type", damage),
                    ("People Count", people_count),
                    ("Time Mentions", time_mentions),
                ]
                for label, values in rows:
                    if values:
                        ctk.CTkLabel(
                            self.actionable_frame,
                            text=f"{label}: {', '.join(values)}",
                            font=ctk.CTkFont(size=12),
                            wraplength=550,
                            justify="left"
                        ).pack(anchor="w", padx=20, pady=2)

                if location_note:
                    ctk.CTkLabel(
                        self.actionable_frame,
                        text=f"Location Note: {location_note}",
                        font=ctk.CTkFont(size=11),
                        text_color="gray",
                        wraplength=550,
                        justify="left"
                    ).pack(anchor="w", padx=20, pady=(4, 2))


class MainDashboard(ctk.CTk):
    """Main dashboard application with HITL Database Integration."""
    
    def __init__(self):
        super().__init__()
        
        self.title("Disaster Tweet Classification Dashboard")
        self.geometry("1400x900")
        
        self.api_client = APIClient()
        self.token_highlighter = TokenHighlighter(min_color="#FFFFFF", max_color="#FF6B6B")
        self.model_inference = None
        
        self.tweets = []
        self.current_filter = None
        self.hitl_mode = False
        self.verified_filter_mode = False
        
        self.setup_ui()
        self.show_connection()
    
    def setup_ui(self):
        self.main_container = ctk.CTkFrame(self)
        self.main_container.pack(fill="both", expand=True)
        
        self.connection_frame = ConnectionFrame(self.main_container, self.on_connection_success)
        self.dashboard_frame = ctk.CTkFrame(self.main_container)
        self.setup_dashboard()
    
    def setup_dashboard(self):
        """Setup dashboard UI with GRID layout."""
        top_bar = ctk.CTkFrame(self.dashboard_frame, height=70)
        top_bar.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(top_bar, text="Disaster Tweet Classifier", font=ctk.CTkFont(size=24, weight="bold")).pack(side="left", padx=20)
        
        # REFRESH DATABASE BUTTON
        self.refresh_btn = ctk.CTkButton(
            top_bar, 
            text="Refresh Database", 
            width=150,
            fg_color="#34495E",
            hover_color="#2C3E50",
            command=self.load_tweets_from_db
        )
        self.refresh_btn.pack(side="right", padx=20)
        
        self.content_area = ctk.CTkFrame(self.dashboard_frame, fg_color="transparent")
        self.content_area.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Grid layout for proportions
        self.content_area.grid_columnconfigure(0, weight=4)
        self.content_area.grid_columnconfigure(1, weight=6)
        self.content_area.grid_rowconfigure(0, weight=1)
        
        # LEFT PANEL
        left_panel = ctk.CTkFrame(self.content_area)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        self.tweet_input_frame = TweetInputFrame(left_panel, self.handle_manual_tweet)
        self.tweet_input_frame.pack(fill="x", padx=10, pady=10)
        
        filter_box = ctk.CTkFrame(left_panel)
        filter_box.pack(fill="x", padx=10, pady=5)
        
        filter_scroll = ctk.CTkScrollableFrame(filter_box, orientation="horizontal", height=50)
        filter_scroll.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkButton(filter_scroll, text="All", width=80, command=lambda: self.filter_tweets(None)).pack(side="left", padx=5)
        
        self.hitl_btn = ctk.CTkButton(
            filter_scroll, text="To Verify", fg_color="#D35400", hover_color="#E67E22",
            command=self.filter_hitl, width=100
        )
        self.hitl_btn.pack(side="left", padx=5)

        self.verified_btn = ctk.CTkButton(
            filter_scroll, text="Human Verified", fg_color="#27AE60", hover_color="#2ECC71",
            command=self.filter_verified, width=130
        )
        self.verified_btn.pack(side="left", padx=5)

        for lid, name in LABEL_DISPLAY_NAMES.items():
            ctk.CTkButton(
                filter_scroll, text=name, width=120, fg_color=LABEL_COLORS.get(lid, "#95A5A6"),
                command=lambda l=lid: self.filter_tweets(l)
            ).pack(side="left", padx=5)
        
        self.tweet_list_frame = ctk.CTkScrollableFrame(left_panel)
        self.tweet_list_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # RIGHT PANEL
        right_panel = ctk.CTkFrame(self.content_area)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        
        # Pass self reference for the Detail Refresh button to work
        self.detail_frame = TweetDetailFrame(right_panel, self.token_highlighter, self.api_client, self)
        self.detail_frame.pack(fill="both", expand=True)

    def show_connection(self):
        self.dashboard_frame.pack_forget()
        self.connection_frame.pack(fill="both", expand=True)
    
    def on_connection_success(self):
        self.connection_frame.pack_forget()
        self.dashboard_frame.pack(fill="both", expand=True)
        self.load_tweets_from_db()

    def filter_hitl(self):
        self.hitl_mode = True
        self.verified_filter_mode = False
        self.current_filter = None
        self.display_tweets()

    def filter_verified(self):
        self.hitl_mode = False
        self.verified_filter_mode = True
        self.current_filter = None
        self.display_tweets()

    def filter_tweets(self, label_id):
        self.hitl_mode = False
        self.verified_filter_mode = False
        self.current_filter = label_id
        self.display_tweets()
    
    def handle_manual_tweet(self, tweet_data):
        def process_thread():
            self.classify_single_tweet(tweet_data)
            self.save_tweet_to_db(tweet_data)
            self.load_tweets_from_db()
        threading.Thread(target=process_thread, daemon=True).start()
    
    def classify_single_tweet(self, tweet_data):
        if not self.model_inference: self.model_inference = ModelInference()
        if not self.model_inference.loaded: return
        
        text = tweet_data.get('text', '')
        result = self.model_inference.classify_tweet(text)
        if result:
            confidence = max(result['confidence_scores'])
            status = 'verified' if confidence >= CONFIDENCE_THRESHOLD else 'unverified'
            
            tweet_data['classification'] = {
                'predictedLabelId': result['predicted_label_id'],
                'predictedLabel': result['predicted_label'],
                'confidenceScores': result['confidence_scores'],
                'classifiedAt': datetime.now().isoformat()
            }
            tweet_data['explanation'] = result.get('explanation', [])
            tweet_data['actionableInfo'] = result.get('actionable_info', {})
            tweet_data['status'] = status
    
    def save_tweet_to_db(self, tweet_data):
        try:
            self.api_client.create_tweet(tweet_data)
        except Exception as e: print(f"DB Error: {e}")
    
    def load_tweets_from_db(self):
        """Fetch all changes from DB and restart filtering/classification."""
        # Visual feedback
        if hasattr(self, 'refresh_btn'):
            self.refresh_btn.configure(text="Syncing...", state="disabled")

        def load_thread():
            success, _, tweets, _ = self.api_client.get_tweets(page=1, limit=MAX_TWEETS_PER_FETCH)
            if success: 
                self.classify_tweets_locally(tweets)
            
            # Reset UI button
            if hasattr(self, 'refresh_btn'):
                self.after(500, lambda: self.refresh_btn.configure(text="Refresh Database", state="normal"))

        threading.Thread(target=load_thread, daemon=True).start()
    
    def classify_tweets_locally(self, tweets):
        """Iterate loaded tweets and update status based on threshold."""
        if not self.model_inference: self.model_inference = ModelInference()

        def sync_tweet_to_db(tweet_obj):
            """Push local tweet classification/actionable updates to backend."""
            if not tweet_obj.get('tweetId') or not tweet_obj.get('classification'):
                return
            threading.Thread(
                target=lambda t=tweet_obj: self.api_client.update_tweet_classification(
                    t['tweetId'],
                    t.get('classification', {}),
                    t.get('explanation', []),
                    t.get('actionableInfo', {}),
                    status=t.get('status', 'unverified')
                ),
                daemon=True
            ).start()
        
        processed = []
        for tweet in tweets:
            if 'status' not in tweet:
                tweet['status'] = 'unverified'

            class_data = tweet.get('classification', {})
            scores = class_data.get('confidenceScores', [])
            
            # Logic for items that haven't been reviewed by human or auto-verified yet
            if tweet['status'] == 'unverified':
                if not scores or class_data.get('predictedLabelId') is None:
                    res = self.model_inference.classify_tweet(tweet.get('text', ''))
                    if res:
                        confidence = max(res['confidence_scores'])
                        status = 'verified' if confidence >= CONFIDENCE_THRESHOLD else 'unverified'
                        
                        tweet['classification'] = {
                            'predictedLabelId': res['predicted_label_id'],
                            'predictedLabel': res['predicted_label'],
                            'confidenceScores': res['confidence_scores']
                        }
                        tweet['explanation'] = res.get('explanation', [])
                        tweet['actionableInfo'] = res.get('actionable_info', {})
                        tweet['status'] = status
                        
                        if status == 'verified':
                            sync_tweet_to_db(tweet)
                else:
                    confidence = max(scores)
                    if confidence >= CONFIDENCE_THRESHOLD:
                        tweet['status'] = 'verified'
                        sync_tweet_to_db(tweet)

            # Backfill/migrate actionable info for already-classified actionable tweets.
            pred_label_id = tweet.get('classification', {}).get('predictedLabelId')
            if pred_label_id in ACTIONABLE_LABELS and self.model_inference.loaded:
                recomputed_actionable = self.model_inference.get_actionable_info(
                    tweet.get('text', ''),
                    pred_label_id
                ) or {}
                recomputed_actionable = _normalize_actionable_info(recomputed_actionable)
                current_actionable = _normalize_actionable_info(tweet.get('actionableInfo') or {})

                if recomputed_actionable != current_actionable:
                    tweet['actionableInfo'] = recomputed_actionable
                    sync_tweet_to_db(tweet)
            
            processed.append(tweet)
        
        self.tweets = processed
        self.after(0, self.display_tweets)

    def display_tweets(self):
        """Display filtered tweets based on current UI state."""
        for widget in self.tweet_list_frame.winfo_children(): widget.destroy()
        
        display_list = self.tweets
        
        if self.hitl_mode:
            display_list = [t for t in self.tweets if t.get('status') == 'unverified']
        elif self.verified_filter_mode:
            display_list = [t for t in self.tweets if t.get('status') == 'human_verified']
        elif self.current_filter is not None:
            # Show Auto-verified OR Human-verified for the specific category
            display_list = [
                t for t in self.tweets 
                if t.get('classification', {}).get('predictedLabelId') == self.current_filter 
                and (t.get('status') == 'verified' or t.get('status') == 'human_verified')
            ]
        
        for tweet in display_list:
            card = TweetCard(self.tweet_list_frame, tweet, tweet.get('classification', {}), lambda t=tweet: self.select_tweet(t))
            card.pack(fill="x", padx=5, pady=5)
    
    def select_tweet(self, tweet):
        self.detail_frame.verification_mode = True 
        self.detail_frame.display_tweet(tweet)


def main():
    app = MainDashboard()
    app.mainloop()


if __name__ == "__main__":
    main()

