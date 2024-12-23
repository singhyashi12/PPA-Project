import streamlit as st
import pandas as pd
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
import numpy as np
import requests
import smtplib

st.title('Internet Book Database📚💗')
st.text('Find your next read here🎀')
df=pd.read_csv('books.csv')

st.image('books.jpg')

st.text('Play this soothing music while you find your next read 🎶')
st.audio("books.mp3")

# Ensure 'Price' is numeric and handle errors
df['Price'] = pd.to_numeric(df['Price'].str.replace(r'[\$,]', '', regex=True), errors='coerce')  # Remove $ and commas
df['Price'] = df['Price'].fillna(df['Price'].median())  # Fill NaN with median price

# Preprocess the Category column for Linear Regression
df['Category_Encoded'] = pd.factorize(df['Category'])[0]

# Set up the Linear Regression model
X = df[['Category_Encoded']]  # Using Category as feature
y = df['Price']  # Target is Price

# Train the model (using the entire dataset here; adjust as needed)
model = LinearRegression()
model.fit(X, y)

# Input field for the user to search for a book
name = st.text_input("Search for a book")
st.write("The book you are searching for is:", name)

# Function to fetch book cover URL from Open Library API
def fetch_book_cover(title):
    try:
        base_url = "http://openlibrary.org/search.json"
        params = {"title": title, "limit": 1}
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data['docs']:
            cover_id = data['docs'][0].get('cover_i')
            if cover_id:
                return f"http://covers.openlibrary.org/b/id/{cover_id}-L.jpg"
        return None
    except Exception as e:
        print(f"Error fetching cover for {title}: {e}")
        return None

# Button to trigger the search
if st.button("Search"):
    if name:  # If a book name is entered
        row = df[df['Title'].str.contains(name.strip(), case=False, na=False)]
        
        if not row.empty:
            details = row.iloc[0]  # Get the first matching book

            # Display book details
            st.write(f"**Title:** {details['Title']}")
            st.write(f"**Author:** {details['Author']}")
            st.write(f"**Description:** {details['Description']}")
            st.write(f"**Category:** {details['Category']}")
            st.write(f"**Publisher:** {details['Publisher']}")
            st.write(f"**Price: $** {details['Price']}")
            st.write(f"**Publish Year:** {details['Published']}")
            
            # Fetch and display book cover image
            cover_url = fetch_book_cover(details['Title'])
            if cover_url:
                st.image(cover_url, width=200)
            else:
                st.write("❌No image available for this book.")

            # Predict the price using the Linear Regression model
            category_encoded = pd.factorize([details['Category']])[0][0]
            predicted_price = model.predict([[category_encoded]])[0]
            #st.write(f"**Predicted Price Range:** ${predicted_price:.2f}")

            # Define the price range (± $5)
            price_min = predicted_price - 5
            price_max = predicted_price + 5

            # Filter similar books based on the price range and category
            similar_books = df[
                (df['Category'] == details['Category']) &  # Same category
                (df['Price'] >= price_min) & 
                (df['Price'] <= price_max) & 
                (df['Title'] != details['Title'])  # Exclude the selected book
            ]

            # Display recommendations without images
            st.write("### Books you might like:")
            if not similar_books.empty:
                for _, book in similar_books.head(10).iterrows():
                    st.write(f"- **{book['Title']}** (${book['Price']})")
            else:
                #st.write("### Books you might like")
                random_books = df[df['Title'] != details['Title']].sample(10)  # Exclude the searched book and get 10 random books
                for _, book in random_books.iterrows():
                    st.write(f"- **{book['Title']}** (${book['Price']})")
        else:
            st.error("No book found with that name!")
    else:
        st.write("Please enter a book name.")

# Displaying Quote
quote_html = """
<div style="
    position: fixed;
    bottom: 10px;
    width: 100%;
    text-align: center;
    font-size: 36px;
    font-style: italic;
    font-weight: bold;
    color: #FFB6C1;
">
    so many books, so little time ~ frank zappa
</div>
"""

# Inject the HTML into Streamlit
st.markdown(quote_html, unsafe_allow_html=True)
