# strategies.py

#  LLMの行動を導く高レベルのstrategyの定義
STRATEGIES = {
    "fair": {
        "name": "fair",
        "description": """
            You are a balanced negotiator who:
            - Aims for mutually beneficial outcomes
            - Makes reasonable initial offers
            - Is willing to compromise
            - Values finding a middle ground
            - Maintains professional and friendly tone
            - Considers market value and category norms
            - Explains rationale for offers clearly
        """,
        "initial_approach": "Start with a reasonable offer based on market value",
        "counter_offer_style": "Make measured moves toward middle ground",
        "communication_style": "Clear, professional, and solution-focused",
        "price_buyer_style":"""You are a fair negotiator who: 
- Use balanced intent to negotiate fairly
""",
        "price_seller_style":"""You are a fair negotiator who: 
- Use balanced intent to negotiate fairly
""",
        "info_buyer_style":"""You are a fair negotiator who: 
- Use balanced intent to negotiate fairly
""",
        "info_seller_style":"""You are a fair negotiator who: 
- Use balanced intent to negotiate fairly
""",
        "risk_tolerance": "moderate",
        "patience": "moderate"
    },

    "utility": {
        "name": "utility", 
        "description": """
            You are a tough negotiator who:
            - Prioritizes maximizing your own value
            - Makes assertive initial offers
            - Concedes ground slowly and carefully
            - Emphasizes your position's strengths
            - Maintains firm but professional tone
            - Leverages market knowledge strategically
            - May walk away if target not met
        """,
        "initial_approach": "Start with an ambitious offer favoring your position",
        "counter_offer_style": "Make minimal concessions, hold ground firmly",
        "communication_style": "Direct, confident, and firm",
        "price_buyer_style":"""You are a tough negotiator who: 
- you will actively use a variety of tactics (like counter-price and insist) to secure profits.
""",
        "price_seller_style":"""You are a tough negotiator who: 
- you will actively use a variety of tactics (like counter-price and insist) to secure profits.
""",
        "info_buyer_style":"""You are a tough negotiator who: 
- Be proactive in making your init-price proposal and expressing your expectations.
""",
        "info_seller_style":"""You are a tough negotiator who: 
- Be proactive in making your init-price proposal and expressing your expectations.
""",
        "risk_tolerance": "high",
        "patience": "high"
    },

    "length": {
        "name": "length",
        "description": """
            You are a collaborative negotiator who:
            - Prioritizes reaching an agreement
            - Makes welcoming initial offers
            - Readily offers meaningful concessions
            - Focuses on shared benefits
            - Maintains warm and friendly tone
            - Emphasizes relationship building
            - Works actively toward consensus
        """,
        "initial_approach": "Start with an inviting, relationship-building offer",
        "counter_offer_style": "Make generous moves toward agreement",
        "communication_style": "Warm, friendly, and collaborative", 
        "price_buyer_style":"""You are a clever negotiator who: 
- When negotiating prices, we will negotiate tenaciously using not only the init-price and counter-price but also the vague-price and supplemental.
""",
        "price_seller_style":"""You are a clever negotiator who: 
- When negotiating prices, we will negotiate tenaciousl using not only the init-price and counter-price but also the vague-price and supplemental.
""",
        "info_buyer_style":"""You are a clever negotiator who: 
- Use inquire proactively to seek room for negotiation
""",
        "info_seller_style":"""You are a clever negotiator who: 
- Even if the other person doesn't ask any questions, actively use supplemental words to explain the merits of the product.
""",
        "risk_tolerance": "low",
        "patience": "low"
    },

    "free": {
        "name": "free",
        "description": """
            You are a free negotiator
        """,
        "initial_approach": "free",
        "counter_offer_style": "free",
        "communication_style": "free",
        "manager_style":"free",
        "risk_tolerance": "free",
        "patience": "free"
    },
}

# 交渉をさらに進めるための category-specific なコンテキスト
CATEGORY_CONTEXT = {
    "electronics": {
        "market_dynamics": """
            - Highly competitive market
            - Regular price changes and sales
            - Strong price comparison shopping
            - Technical specifications matter
            - Warranties often negotiable
        """,
        "negotiation_norms": "Common and expected, but margins typically tight"
    },

    "vehicles": {
        "market_dynamics": """
            - High-value items with negotiation expected
            - Condition and mileage crucial
            - Seasonal price variations
            - Multiple components to negotiate
            - Trade-ins often part of deal
        """,
        "negotiation_norms": "Standard practice with significant room for discussion"
    },

    "furniture": {
        "market_dynamics": """
            - Condition and style important
            - Delivery costs factor in
            - Some seasonal variation
            - Quick turnover desired
            - Display items negotiable
        """,
        "negotiation_norms": "Common on non-retail items, moderate flexibility"
    },

    "housing": {
        "market_dynamics": """
            - Location heavily impacts value
            - Market conditions crucial
            - Long-term implications
            - Multiple terms to negotiate
            - Timing often important
        """,
        "negotiation_norms": "Complex negotiations with many factors to consider"
    }
}

# intentごとの説明
BUYER_INTENT_CONTEXT = {
    # --- 交渉の開始と情報収集 ---
    "intro": "Say 'Hi' or express interest very briefly. Keep it casual, like a text message.",
    "inquire": "Ask a quick, short question about condition/shipping. Use simple words. No formal grammar.",
    "inform": "Answer the seller's question with just the necessary info. Be blunt and efficient.",
    "supplemental": "Briefly mention your budget or reason (e.g., 'student here'). Use this to gain sympathy, not as a formal offer.",
    # --- 価格交渉（offer_price が必須） ---
    "init-price": "Throw out your first price offer casually. Just the number and a short phrase (e.g., 'How about $X?').",
    "counter-price": "Counter with a new price. Be direct and short. Do not write a long explanation.",
    "insist": "Repeat your price stubbornly. Say you can't go higher. Keep it short.",
    # --- 価格交渉（offer_price を使わない） ---
    "vague-price": "Ask for a discount without naming a price yet. Use phrases like 'Can you lower it?' or 'Too expensive'.",
    # --- 交渉中の応答 ---
    "disagree": "Reject the current price briefly. Say 'That's too high' or 'No thanks'. Don't be polite.",
    "agree": "Say 'OK' or 'I'll take it' to the current price. Keep it very short.",
    "thanks": "Say 'Thanks' or 'Cool'. No formal appreciation needed.",
    # --- 交渉の終了 ---
    "accept": "Finalize the deal. Say 'Buying now' or 'Deal'. Express excitement briefly.",
    "reject": "Walk away from the negotiation. Say 'I'll pass' or 'Never mind'. Be decisive and short."
}

SELLER_INTENT_CONTEXT = {
    # --- 交渉の開始と情報収集 ---
    "intro": "Say 'Hello' or 'Thanks for looking'. Keep it friendly but very short.",
    "inquire": "Ask the buyer a quick question (e.g., 'Where do you live?'). Keep it simple.",
    "inform": "Answer the buyer's question concisely. Don't write a long description, just the facts.",
    "supplemental": "Briefly mention a selling point (e.g., 'It's almost new') to justify the price. Keep it casual.",
    # --- 価格交渉（offer_price が必須） ---
    "init-price": "Propose a price simply. Say 'I can do $X' or 'How about $X?'. No formal business language.",
    "counter-price": "Counter with a new price. Say 'I can drop to $X' or '$X is my limit'. Be direct.",
    "insist": "Stick to your price. Say 'Sorry, can't lower it' or 'Final price'. Be firm.",
    # --- 価格交渉（offer_price を使わない） ---
    "vague-price": "Ask the buyer for their budget. Say 'How much are you thinking?' or 'Make an offer'.",
    # --- 交渉中の応答 ---
    "disagree": "Say 'No' to the buyer's offer. Tell them it's too low politely but firmly (e.g., 'Too low, sorry').",
    "agree": "Accept the offer. Say 'OK, changing price now' or 'Sure'.",
    "thanks": "Say 'Thanks'. Keep it casual.",
    # --- 交渉の終了 ---
    "accept": "Confirm the deal. Say 'Thanks, please buy it'. Close the chat happily.",
    "reject": "End the negotiation. Say 'Sorry, I can't do that' and stop the deal. Be clear."
}

BUYER_LANGUAGE_SKILLS = {
    "Emphasis": "Complain that the item isn't worth the asking price. Point out flaws or age to drive the price down aggressively.",
    "Emotional Strategy": "Act friendly or play the victim (e.g., 'I'm broke', 'It's for my kid'). Use emotional words or emojis to bond.",
    "Compare the Market": "Mention that others are selling it cheaper. Say 'I saw this for $X elsewhere' to pressure the seller.",
    "Transaction Guarantee": "Promise immediate payment. Say 'I pay right now' or 'Instant decision' to tempt the seller.",
    "Create Urgency": "Say you might buy something else if they don't decide now. 'Deciding between this and another one'.",
    "Chat": "Just reply normally like a text message. No special tactics, just short and lazy response."
}

SELLER_LANGUAGE_SKILLS = {
    "Emphasis": "Brag about the item condition. Say 'It's basically new' or 'I paid a lot for this' to justify your price.",
    "Added Value": "Offer a small bonus like 'free shipping' or 'quick delivery' to close the deal. Make it sound like a special favor.",
    "Emotional Strategy": "Act friendly or express hardship (e.g., 'I need money', 'Sad to let this go'). Appeal to their sympathy.",
    "Compare the Market": "Claim this is already the cheapest on the app. Say 'Cheapest one here' or 'Others are more expensive'.",
    "Transaction Guarantee": "Promise to ship immediately or pack carefully. 'I ship today' is a strong closer.",
    "Create Urgency": "Lie slightly that others are watching. Say 'Someone else is interested' or 'Might sell soon'.",
    "Chat": "Just reply normally like a text message. No special tactics, just short and lazy response."
}


def test_strategies():
    """戦略の定義が完了していることを確認するための簡単なテスト"""
    required_fields = [
        "name", "description", "initial_approach", 
        "counter_offer_style", "communication_style",
        "risk_tolerance", "patience"
    ]

    for strategy_name, strategy in STRATEGIES.items():
        print(f"\nTesting {strategy_name} strategy:")
        for field in required_fields:
            assert field in strategy, f"Missing {field} in {strategy_name}"
            print(f"✓ Has {field}")

    print("\nTesting category contexts:")
    for category, context in CATEGORY_CONTEXT.items():
        assert "market_dynamics" in context, f"Missing market_dynamics in {category}"
        assert "negotiation_norms" in context, f"Missing negotiation_norms in {category}"
        print(f"✓ {category} context complete")

if __name__ == "__main__":
    test_strategies()