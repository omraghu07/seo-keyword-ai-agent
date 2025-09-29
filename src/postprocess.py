# src/postprocess.py
"""
Post-processing tool for keyword research results
Cleans, annotates, and formats CSV output for professional presentation
"""

import pandas as pd
from datetime import date, datetime
import os
import re
import json

# Install these if you haven't: pip install pandas openpyxl tabulate
try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False
    print("Note: Install 'tabulate' for prettier table output: pip install tabulate")

try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    print("Note: Install 'openpyxl' for Excel export: pip install openpyxl")

# Configuration
BRAND_KEYWORDS = {
    "linkedin", "indeed", "glassdoor", "ucla", "asu", "berkeley", 
    "hennge", "ciee", "google", "facebook", "microsoft", "amazon",
    "apple", "netflix", "spotify", "youtube", "instagram", "twitter"
}
OUTPUT_DIR = "results"  # Directory to save processed files

def normalize_keyword(keyword):
    """Clean and normalize keyword text"""
    if not keyword or pd.isna(keyword):
        return ""
    return str(keyword).strip()

def is_brand_query(keyword, brand_set=BRAND_KEYWORDS):
    """
    Check if keyword is a brand/navigational query
    These are harder to rank for if you're not that brand
    """
    if not keyword:
        return False
    
    keyword_lower = keyword.lower()
    
    # Check if any brand name appears in keyword
    for brand in brand_set:
        if brand in keyword_lower:
            return True
    
    # Check for domains (.com, .edu, etc.)
    if re.search(r"\.(com|edu|org|net|gov|io)\b", keyword_lower):
        return True
    
    return False

def classify_search_intent(keyword):
    """
    Classify keyword by search intent:
    - informational: seeking information
    - commercial: researching before buying
    - transactional: ready to take action
    - navigational: looking for specific site/brand
    """
    if not keyword:
        return "informational"
    
    keyword_lower = keyword.lower()
    
    # Informational intent signals
    if any(signal in keyword_lower for signal in [
        "how to", "what is", "why", "are", "do ", "does ", "can ", 
        "guide", "tutorial", "learn", "definition", "meaning"
    ]):
        return "informational"
    
    # Transactional intent signals
    if any(signal in keyword_lower for signal in [
        "buy", "price", "cost", "apply", "register", "admission", 
        "apply now", "enroll", "join", "signup", "book", "order"
    ]):
        return "transactional"
    
    # Commercial intent signals
    if any(signal in keyword_lower for signal in [
        "best", "top", "compare", "vs", "reviews", "review", 
        "cheap", "affordable", "discount", "deal"
    ]):
        return "commercial"
    
    # Navigational intent (brand queries)
    if is_brand_query(keyword):
        return "navigational"
    
    # Default to informational
    return "informational"

def classify_keyword_tail(keyword):
    """
    Classify keyword by tail length:
    - short-tail: 1-2 words (high competition, high volume)
    - mid-tail: 3 words (moderate competition/volume)
    - long-tail: 4+ words (low competition, low volume)
    """
    if not keyword:
        return "short-tail"
    
    word_count = len(str(keyword).split())
    
    if word_count >= 4:
        return "long-tail"
    elif word_count == 3:
        return "mid-tail"
    else:
        return "short-tail"

def format_large_number(number):
    """Format large numbers with commas for readability"""
    try:
        return f"{int(number):,}"
    except (ValueError, TypeError):
        return str(number)

def clean_and_process_dataframe(df, seed_keyword):
    """Main processing function to clean and enhance the dataframe"""
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    print("🧹 Cleaning and processing data...")
    
    # 1. Normalize keywords and remove duplicates
    df["Keyword"] = df["Keyword"].astype(str).apply(normalize_keyword)
    
    # Remove empty keywords
    df = df[df["Keyword"].str.len() > 0]
    
    # Sort by Opportunity Score and remove duplicates (keep highest score)
    df = df.sort_values(by="Opportunity Score", ascending=False)
    df = df.drop_duplicates(subset=["Keyword"], keep="first")
    
    # 2. Fix data types and handle missing values
    
    # Monthly Searches: convert to int, fill missing with 0
    df["Monthly Searches"] = pd.to_numeric(df["Monthly Searches"], errors="coerce").fillna(0).astype(int)
    
    # Competition: round to 4 decimal places
    df["Competition"] = pd.to_numeric(df["Competition"], errors="coerce").fillna(0.0).round(4)
    
    # Opportunity Score: round to 2 decimal places for readability
    df["Opportunity Score"] = pd.to_numeric(df["Opportunity Score"], errors="coerce").fillna(0.0).round(2)
    
    # Google Results: clean and convert to int
    if "Google Results" in df.columns:
        # Remove any non-digit characters and convert to int
        df["Google Results"] = df["Google Results"].astype(str).str.replace(r"[^\d]", "", regex=True)
        df["Google Results"] = pd.to_numeric(df["Google Results"], errors="coerce").fillna(0).astype(int)
    
    # Ads Shown: convert to int
    if "Ads Shown" in df.columns:
        df["Ads Shown"] = pd.to_numeric(df["Ads Shown"], errors="coerce").fillna(0).astype(int)
    
    # 3. Add enhancement columns
    print("📊 Adding analysis columns...")
    
    df["Intent"] = df["Keyword"].apply(classify_search_intent)
    df["Tail"] = df["Keyword"].apply(classify_keyword_tail)
    df["Is Brand/Navigational"] = df["Keyword"].apply(lambda x: "Yes" if is_brand_query(x) else "No")
    
    # 4. Reorder columns for better presentation
    column_order = [
        "Keyword",
        "Intent", 
        "Tail",
        "Is Brand/Navigational",
        "Monthly Searches",
        "Competition", 
        "Opportunity Score",
        "Google Results",
        "Ads Shown",
        "Featured Snippet?",
        "PAA Available?",
        "Knowledge Graph?"
    ]
    
    # Only include columns that exist in the dataframe
    available_columns = [col for col in column_order if col in df.columns]
    df = df[available_columns]
    
    # 5. Final sort by Opportunity Score
    df = df.sort_values(by="Opportunity Score", ascending=False).reset_index(drop=True)
    
    print(f"✅ Processing complete! {len(df)} keywords ready")
    return df

def save_processed_results(df, seed_keyword, output_dir=OUTPUT_DIR):
    """Save processed results in multiple formats with metadata"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate safe filename from seed keyword
    today = date.today().isoformat()
    safe_seed = re.sub(r"[^\w\s-]", "", seed_keyword).strip().replace(" ", "_")[:50]
    base_filename = f"keywords_{safe_seed}_{today}"
    
    # File paths
    csv_path = os.path.join(output_dir, f"{base_filename}.csv")
    excel_path = os.path.join(output_dir, f"{base_filename}.xlsx")
    meta_path = os.path.join(output_dir, f"{base_filename}.meta.json")
    
    # Save CSV
    df.to_csv(csv_path, index=False)
    print(f"💾 Saved CSV: {csv_path}")
    
    # Save Excel with multiple sheets (if openpyxl is available)
    if EXCEL_AVAILABLE:
        try:
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                # Top 50 sheet
                df.head(50).to_excel(writer, sheet_name="Top_50", index=False)
                # All results sheet
                df.to_excel(writer, sheet_name="All_Keywords", index=False)
                # Summary sheet
                summary_data = {
                    "Metric": [
                        "Total Keywords", 
                        "Informational Keywords",
                        "Commercial Keywords", 
                        "Transactional Keywords",
                        "Navigational Keywords",
                        "Long-tail Keywords",
                        "Brand/Navigational Keywords"
                    ],
                    "Count": [
                        len(df),
                        len(df[df["Intent"] == "informational"]),
                        len(df[df["Intent"] == "commercial"]),
                        len(df[df["Intent"] == "transactional"]),
                        len(df[df["Intent"] == "navigational"]),
                        len(df[df["Tail"] == "long-tail"]),
                        len(df[df["Is Brand/Navigational"] == "Yes"])
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
            
            print(f"📊 Saved Excel: {excel_path}")
        except Exception as e:
            print(f"⚠️ Could not save Excel file: {e}")
    else:
        print("📊 Excel export skipped (install openpyxl to enable)")
    
    # Save metadata
    metadata = {
        "seed_keyword": seed_keyword,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_keywords": len(df),
        "data_source": "SerpApi with heuristic search volumes",
        "methodology": "Opportunity Score = log10(volume+1) / (competition + 0.01)",
        "notes": [
            "Brand/navigational queries are flagged for filtering",
            "Search volumes are estimated - replace with real API data for production",
            "Competition scores based on SERP feature analysis"
        ],
        "intent_breakdown": {
            "informational": int(len(df[df["Intent"] == "informational"])),
            "commercial": int(len(df[df["Intent"] == "commercial"])),
            "transactional": int(len(df[df["Intent"] == "transactional"])),
            "navigational": int(len(df[df["Intent"] == "navigational"]))
        },
        "tail_breakdown": {
            "short-tail": int(len(df[df["Tail"] == "short-tail"])),
            "mid-tail": int(len(df[df["Tail"] == "mid-tail"])),
            "long-tail": int(len(df[df["Tail"] == "long-tail"]))
        }
    }
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"📋 Saved metadata: {meta_path}")
    
    return csv_path, excel_path, meta_path

def display_results_preview(df, top_n=10):
    """Display a nice preview of the top results"""
    
    if df.empty:
        print("❌ No results to display!")
        return
    
    print(f"\n🏆 Top {min(top_n, len(df))} Keywords:")
    
    # Prepare data for display
    preview_df = df.head(top_n).copy()
    
    # Format large numbers for readability
    if "Monthly Searches" in preview_df.columns:
        preview_df["Monthly Searches"] = preview_df["Monthly Searches"].apply(format_large_number)
    
    if "Google Results" in preview_df.columns:
        preview_df["Google Results"] = preview_df["Google Results"].apply(format_large_number)
    
    # Display using tabulate if available
    if TABULATE_AVAILABLE:
        print(tabulate(preview_df, headers="keys", tablefmt="github", showindex=False))
    else:
        # Fallback display
        for i, row in preview_df.iterrows():
            print(f"{i+1}. {row['Keyword']} | Score: {row['Opportunity Score']} | "
                  f"Volume: {row['Monthly Searches']} | Competition: {row['Competition']} | "
                  f"Intent: {row['Intent']} | Tail: {row['Tail']}")

def postprocess_keywords(csv_file_path, seed_keyword):
    """
    Main postprocessing function
    Call this after your ranking.py generates the initial CSV
    """
    
    print(f"🚀 Starting postprocessing for: '{seed_keyword}'")
    print(f"📁 Input file: {csv_file_path}")
    
    try:
        # Load the CSV from ranking.py
        df = pd.read_csv(csv_file_path)
        print(f"📊 Loaded {len(df)} keywords from CSV")
        
        # Clean and process the data
        processed_df = clean_and_process_dataframe(df, seed_keyword)
        
        # Save in multiple formats
        csv_path, excel_path, meta_path = save_processed_results(processed_df, seed_keyword)
        
        # Display preview
        display_results_preview(processed_df, top_n=10)
        
        # Summary stats
        print(f"\n📈 Summary Statistics:")
        print(f"• Total keywords analyzed: {len(processed_df)}")
        print(f"• Long-tail opportunities: {len(processed_df[processed_df['Tail'] == 'long-tail'])}")
        print(f"• Non-brand keywords: {len(processed_df[processed_df['Is Brand/Navigational'] == 'No'])}")
        print(f"• High opportunity (score > 50): {len(processed_df[processed_df['Opportunity Score'] > 50])}")
        
        return csv_path, excel_path, meta_path, processed_df
        
    except Exception as e:
        print(f"❌ Error during postprocessing: {e}")
        raise

# Example usage
if __name__ == "__main__":
    # Example: process a CSV file generated by ranking.py
    input_csv = "best_keywords_2025-09-23.csv"  # Replace with your actual file
    seed_keyword = "global internship"
    
    if os.path.exists(input_csv):
        postprocess_keywords(input_csv, seed_keyword)
    else:
        print(f"❌ Input file not found: {input_csv}")
        print("Run your ranking.py script first to generate the initial CSV")