import streamlit as st
import newspaper
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from urllib.parse import urlparse

# Initialize session state for model and tokenizer
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None

@st.cache_resource
def load_model():
    try:
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the model
        model = T5ForConditionalGeneration.from_pretrained('t5-base')
        
        # Load the saved weights with appropriate map_location
        checkpoint = torch.load('abstractive-model-sihanas.pth', map_location=device)
        
        model.load_state_dict(checkpoint)
        model.to(device)
        
        # Load tokenizer
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        
        return model, tokenizer, device
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def clean_text(text):
    """Clean and preprocess the input text"""
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove very long words (likely garbage)
    text = ' '.join(word for word in text.split() if len(word) < 100)
    return text

def summarize_text(text, model, tokenizer, device):
    try:
        # Clean the text
        cleaned_text = clean_text(text)
        
        # Tokenize and generate summary
        inputs = tokenizer.encode("summarize: " + cleaned_text, 
                                  return_tensors='pt', 
                                  max_length=512, 
                                  truncation=True).to(device)
        
        summary_ids = model.generate(
            inputs,
            max_length=150,
            min_length=40,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    except Exception as e:
        st.error(f"Error in summarization: {str(e)}")
        return None

def fetch_article(url):
    """Fetch article content and metadata from URL using newspaper3k"""
    try:
        # Download and parse the article
        article = newspaper.Article(url)
        
        # Enable extraction of all possible metadata
        article.download()
        article.parse()
        
        # Extract metadata
        title = article.title or 'No title found'
        authors = ', '.join(article.authors) if article.authors else 'No author information'
        publish_date = article.publish_date or 'No publish date found'
        
        # Extract publisher from URL domain
        publisher = urlparse(url).netloc.replace('www.', '').capitalize() or 'No publisher information'
        
        # Get the main text content
        text = article.text or ''
        
        return title, authors, str(publish_date), publisher, text
    
    except Exception as e:
        st.error(f"Error fetching the article: {str(e)}")
        return None, None, None, None, None
    
def main():
    st.title("News Article Summarizer")
    st.write("Enter a news article URL to get a summary.")
    
    # Load model and tokenizer
    model, tokenizer, device = load_model()
    
    if model is None or tokenizer is None:
        st.error("Failed to load the model. Please check your model file and dependencies.")
        return
    
    # URL input
    url = st.text_input("News Article URL")
    
    if st.button("Summarize"):
        if not url:
            st.warning("Please enter a URL")
            return
            
        with st.spinner("Fetching article and generating summary..."):
            # Fetch article
            title, authors, publish_date, publisher, article_text = fetch_article(url)
            
            if article_text:
                # Display metadata
                st.write(f"**Title**: {title}")
                st.write(f"**Authors**: {authors}")
                st.write(f"**Publish Date**: {publish_date}")
                st.write(f"**Publisher**: {publisher}")
                
                # Generate summary
                summary = summarize_text(article_text, model, tokenizer, device)
                
                if summary:
                    st.success("Summary generated successfully!")
                    st.write("### Summary")
                    st.write(summary)
                    
                    # Display original text (collapsed)
                    with st.expander("Show original article"):
                        st.write(article_text)
            else:
                st.error("Failed to fetch the article. Please check the URL and try again.")

if __name__ == "__main__":
    main()