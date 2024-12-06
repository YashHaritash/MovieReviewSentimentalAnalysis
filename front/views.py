from django.shortcuts import render
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Load the model and tokenizer at server start
model_path = 'transformer_model'  # Path to your pre-trained model
model = AutoModelForSequenceClassification.from_pretrained(model_path, from_tf=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)
sentiment_analyzer = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Sample data of movies and reviews
movie_reviews_data = {
    "The Matrix": [
        "This movie redefined sci-fi! An absolute classic.",
        "Mind-blowing concept and stunning visuals.",
        "Keanu Reeves did an incredible job as Neo!",
        "The storyline was confusing at first but ultimately brilliant.",
        "The action sequences are top-notch.",
        "A bit overrated, but still worth a watch.",
        "I didn't really enjoy the plot, felt too slow.",
        "Special effects were great, but the story was lacking.",
        "The concept is unique, but the execution could be better.",
        "Amazing movie with a powerful message.",
        "The sequels didn't live up to this original masterpiece.",
        "Boring at times, but still innovative for its time.",
        "Loved the philosophical undertones.",
        "Too much action, not enough substance for my taste.",
        "A groundbreaking film in every sense.",
        "Didn't quite understand the hype.",
        "One of the best movies of the 90s!",
        "Visually stunning, but the plot could have been simpler.",
        "Revolutionary! It changed the film industry forever.",
        "Not my favorite, but I can see why others like it."
    ],
    "Inception": [
        "A true masterpiece by Christopher Nolan.",
        "Keeps you guessing until the very end.",
        "Mind-bending plot with a phenomenal cast.",
        "Complex but incredibly rewarding to watch.",
        "The concept of dreams within dreams is fascinating.",
        "Too confusing for me, lost interest halfway through.",
        "The visuals are amazing, but the story is even better.",
        "Not a fan of the ambiguous ending.",
        "Great movie, but you need to watch it multiple times to understand.",
        "An intense, unforgettable experience.",
        "Loved every bit of it!",
        "Nolan's direction is superb.",
        "A bit overrated, but still worth watching.",
        "The sound design is just incredible.",
        "Fascinating idea, but hard to follow.",
        "A real brain-twister.",
        "Too complicated for my liking.",
        "A solid performance by Leonardo DiCaprio.",
        "Masterfully crafted movie!",
        "Felt a bit too long, but still great."
    ],
    "The Room": [
        "Worst movie I've ever seen.",
        "The acting is terrible, laughable at best.",
        "Why does this movie even exist?",
        "Plot makes no sense at all.",
        "A cringe-worthy experience from start to finish.",
        "Not even so-bad-it's-good, just bad.",
        "Avoid this movie at all costs.",
        "Painfully boring.",
        "Waste of time, couldn't finish it.",
        "So bad it has become a joke.",
        "Unintentionally hilarious.",
        "One of the most confusing movies ever made.",
        "Acting is absolutely horrendous.",
        "Would not recommend it to anyone.",
        "Why is this even classified as a movie?"
    ],
    "The Great Gatsby": [
        "Amazing visuals and costume design.",
        "Loved Leonardo's portrayal of Gatsby.",
        "A bit style over substance, but still enjoyable.",
        "The story felt hollow.",
        "Music choice was odd for a period film.",
        "Some parts felt unnecessarily dragged out.",
        "Interesting adaptation, but lacks the depth of the book.",
        "Beautiful cinematography.",
        "A mixed bag, not bad but not great either.",
        "Visually stunning but emotionally flat."
    ]
}

# Function to fetch reviews from predefined data and perform sentiment analysis
# Function to fetch reviews from predefined data and perform sentiment analysis
def get_reviews(movie_name):
    reviews = movie_reviews_data.get(movie_name)

    if not reviews:
        return [], "Movie not found", 0, 0  # Ensure 4 values are returned if movie is not found

    # Initialize counts for sentiment analysis
    positive_count = 0
    negative_count = 0
    reviews_text = []

    # Analyze reviews
    for review in reviews:
        sentiment_result = sentiment_analyzer(review)[0]
        
        # Convert label to POSITIVE/NEGATIVE based on model output
        sentiment_label = sentiment_result['label']
        if sentiment_label == 'LABEL_1':
            sentiment_label = 'POSITIVE'
            positive_count += 1
        else:
            sentiment_label = 'NEGATIVE'
            negative_count += 1

        reviews_text.append((review, sentiment_label))

    # Determine overall sentiment
    overall_sentiment = "Good" if positive_count > negative_count else "Bad"
    
    return reviews_text, overall_sentiment, positive_count, negative_count


# Django view to handle movie name input and show results
def greet_view(request):
    movie_name = None
    reviews_text = []
    overall_sentiment = None
    positive_count = 0
    negative_count = 0

    if request.method == 'POST':
        movie_name = request.POST.get('movie_name')  # Get movie name from the form
        if movie_name:
            reviews_text, overall_sentiment, positive_count, negative_count = get_reviews(movie_name)
            print("Reviews fetched:", reviews_text)
    
    return render(request, 'front.html', {
        'movie_name': movie_name,
        'reviews': reviews_text,
        'overall_sentiment': overall_sentiment,
        'positive_count': positive_count,
        'negative_count': negative_count
    })
