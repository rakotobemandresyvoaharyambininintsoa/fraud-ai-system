import pandas as pd
import random


countries = [
    "Madagascar",
    "USA",
    "France",
    "Nigeria",
    "Japan",
    "UK",
    "Germany"
]

times = [
    "morning",
    "afternoon",
    "evening",
    "night"
]

devices = [
    "mobile",
    "desktop",
    "tablet"
]

merchants = [
    "amazon",
    "uber",
    "shopify",
    "unknown",
    "bank"
]


def generate_transactions(n=1000):

    transactions = []

    for user_id in range(1001, 1001+n):

        amount = random.randint(50, 10000)

        country = random.choice(countries)
        time = random.choice(times)
        device = random.choice(devices)
        merchant = random.choice(merchants)

        # logique simple de fraude
        fraud = 0

        if amount > 5000:
            fraud = 1

        if merchant == "unknown":
            fraud = 1

        if time == "night" and device == "desktop":
            fraud = 1

        transactions.append({
            "user_id": user_id,
            "amount": amount,
            "country": country,
            "time": time,
            "device": device,
            "merchant": merchant,
            "is_fraud": fraud
        })

    return pd.DataFrame(transactions)


if __name__ == "__main__":

    df = generate_transactions(1000)

    df.to_csv(
        "data/transactions.csv",
        index=False
    )

    print("Dataset généré avec succès")
    print(df.head())