"""
Génère un jeu de données synthétique de transactions bancaires.

Corrections apportées par rapport à la version précédente :
  - Le pays n'utilise plus de noms de pays réels dans la logique de risque
    (aucun pays réel n'est codé comme "intrinsèquement risqué" — ce n'est
    factuellement pas défendable et pose un problème éthique évident).
    Les pays réels ne servent que de valeurs plausibles côté données ; le
    risque géographique est modélisé via un indicateur générique
    "pays_a_risque" séparé, appliqué à des codes fictifs.
  - Le label de fraude n'est plus un simple seuil déterministe (amount > 5000)
    — cela rendait le problème trivialement séparable, d'où les métriques
    à 1.0 partout dans la version précédente (signe de fuite de données /
    problème trop facile, pas une performance réelle). Le label est
    maintenant probabiliste : les transactions à risque ont une probabilité
    plus élevée d'être frauduleuses, pas une certitude, avec du bruit.

Usage : python data/generate_data.py
"""

import os

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

N_TRANSACTIONS = 5000

PAYS_USUELS = ["Madagascar", "France", "USA", "Japan", "UK", "Germany"]
PAYS_A_RISQUE = ["XX", "YY", "ZZ"]  # codes fictifs, pas de pays réel
TIMES = ["morning", "afternoon", "evening", "night"]
DEVICES = ["mobile", "desktop", "tablet"]
MERCHANTS = ["amazon", "uber", "shopify", "bank", "unknown"]


def generer_transactions(n: int) -> pd.DataFrame:
    amount = RNG.gamma(shape=2.0, scale=1200, size=n) + 50

    country = np.where(
        RNG.random(n) < 0.08,  # ~8% des transactions viennent d'un pays à risque
        RNG.choice(PAYS_A_RISQUE, size=n),
        RNG.choice(PAYS_USUELS, size=n),
    )
    time = RNG.choice(TIMES, size=n)
    device = RNG.choice(DEVICES, size=n)
    merchant = RNG.choice(MERCHANTS, size=n, p=[0.25, 0.2, 0.2, 0.25, 0.1])

    # Score de risque latent : combine plusieurs facteurs de façon additive,
    # avec du bruit — pas un seuil unique et déterministe. C'est ce qui rend
    # le problème réaliste (les facteurs de risque se cumulent, aucun n'est
    # une preuve certaine à lui seul).
    risque = (
        0.35 * (amount > np.quantile(amount, 0.85))
        + 0.30 * np.isin(country, PAYS_A_RISQUE)
        + 0.20 * (merchant == "unknown")
        + 0.15 * ((time == "night") & (device == "desktop"))
        + RNG.normal(0, 0.12, size=n)  # bruit : aucun facteur n'est jamais certain
    )

    seuil = np.quantile(risque, 0.97)  # ~3% de fraude, réaliste pour de la fraude bancaire
    is_fraud = (risque > seuil).astype(int)

    return pd.DataFrame(
        {
            "amount": amount.round(2),
            "country": country,
            "time": time,
            "device": device,
            "merchant": merchant,
            "is_fraud": is_fraud,
        }
    )


def main():
    df = generer_transactions(N_TRANSACTIONS)

    out_path = os.path.join(os.path.dirname(__file__), "transactions.csv")
    df.to_csv(out_path, index=False)

    print(f"{len(df)} transactions générées -> {out_path}")
    print(f"Taux de fraude réel : {df['is_fraud'].mean():.2%}")
    print(df.head())


if __name__ == "__main__":
    main()
