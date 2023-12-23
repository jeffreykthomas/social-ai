import numpy as np
import time
import json
import os

import openai

openai.api_key = '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'


def create_characteristics():
    characteristics = {
        'Attractive': 0,
        'Fit': 0,
        'Healthy': 0,
        'Stable': 0,
        'Careful': 0,
        'Accepting': 0,
        'Fun': 0,
        'Funny': 0,
        'Considerate': 0,
        'Powerful': 0,
        'Flexible': 0,
        'Competent': 0,
        'Confident': 0,
        'Honest': 0,
        'Empathic': 0,
        'Humble': 0,
        'Well-trained': 0,
        'Educated': 0,
        'Disciplined': 0,
        'Adaptable': 0,
        'Resilient': 0,
        'Resourceful': 0,
        'Creative': 0,
        'Stylish': 0,
        'Knowledgeable': 0,
        'Happy': 0,
        'Famous': 0,
        'Athletic': 0,
        'Intentional': 0,
        'Mechanical': 0,
        'Diligent': 0,
        'Clever': 0,
        'Organized': 0,
        'Persistent': 0,
        'Efficient': 0,
        'Content': 0,
        'Free': 0,
        'Independent': 0,
        'Reliable': 0,
        'Relief': 0,
        'Respectful': 0,
        'Compliant': 0,
        'Cooperative': 0,
        'Kind': 0,
        'Generous': 0,
        'Focused': 0,
        'Merciful': 0,
        'Responsible': 0,
        'Safe': 0,
        'Encouraging': 0,
        'Constructive': 0,
        'Adventurous': 0,
        'Orderly': 0,
        'Approachable': 0,
        'Responsive': 0,
        'Upfront': 0
    }
    char_str = json.dumps(characteristics, indent=4)
    favorites = {
        'Breakfast': '',
        'Lunch': '',
        'Snack': '',
        'Dinner': '',
        'Dessert': '',
        'TV Show': '',
        'YouTuber': '',
        'Person': '',
        'Family Member': '',
        'Video Game': '',
        'Athlete': '',
        'Sports Team': '',
        'Entertainer': '',
        'Artist': '',
        'Musician': '',
        'Historical Figure': '',
        'Teacher': '',
        'Coach': '',
        'Actor': '',
        'Superhero': '',
        'Song': '',
        'Movie': ''
    }
    fav_str = json.dumps(favorites, indent=4)
    skills = {
        'Energy': 0,
        'Health': 0,
        'Specific Skills': 0,
        'Resilience': 0,
        'Flexibility': 0,
        'Understanding': 0,
        'Willingness': 0,
        'Motivation': 0,
        'Vision': 0,
        'Planning': 0,
        'Patience': 0,
        'Discernment': 0,
        'Humility': 0,
        'Faith': 0,
        'Consistency': 0,
        'Focus': 0,
        'Versatility': 0,
        'Assertion': 0,
        'Adaptability': 0,
        'Endurance': 0
    }
    skill_str = json.dumps(skills, indent=4)
    behaviors = {
        'Avoidance': 0,
        'Low Exertion': 0,
        'Acted Incapable': 0,
        'Acted Afraid': 0,
        'Procrastinated': 0,
        'Indecision': 0,
        'Apathy': 0,
        'Laziness': 0,
        'Overly Dependent': 0,
        'Overly Permissive': 0,
        'Excessive Guilt': 0,
        'Excessive Fear': 0,
        'Excessive Stress': 0,
        'Excessive Anger': 0,
        'Excessive Grief': 0,
        'Self-Deception': 0,
        'Complaint': 0,
        'Excuses': 0,
        'Blaming': 0,
        'Conflicted': 0,
        'Entitlement': 0,
        'Threatening': 0,
        'Striking': 0,
        'Yelling': 0,
        'Badgering': 0,
        'Name Calling': 0,
        'Restricting': 0,
        'Taking': 0,
        'Interrupting': 0,
        'Punishing': 0,
        'Acting Incapable': 0,
        'Acting Afraid': 0,
        'Procrastinating': 0,
        'Indecisiveness': 0,
        'Manipulation': 0,
        'Deception': 0,
        'Making Excuses': 0,
        'Complaining': 0,
        'Worrying': 0,
        'Ignoring': 0,
        'Projecting Negativity': 0,
        'Inefficiency': 0
    }
    behave_str = json.dumps(behaviors, indent=4)

    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": """
            You are a helpful assistant. Your role is to create a coherent fictional character. 
            For each object key you need to change the value to match the fictional character, 
            either on a scale of 1-10 for numeric values, or with the string needed to complete 
            the profile. 
            For all empty lists, fill in a list of at least 5 strings that completes the profile. 
            You should only return a json response with all the required information.
            """
             },
            {"role": "user", "content": f"""
            Here are the objects and lists: 
            {{
            "characteristics": {char_str},
            "favorites": {fav_str},
            "behaviors": {behave_str},
            "skills": {skill_str},
            "preferred_tasks": [],
            "prioritized_activities": [],
            "short_term_goals": [],
            "medium_term_goals": [],
            "long_term_goals": [],
            "mistakes": [],
            "wrong_doings": [],
            "successes": [],
            }}
            """
             },
        ],
    )

    identity = json.loads(response['choices'][0]['message']['content'])
    # Get a list of all files in the current directory
    files = os.listdir('.')

    # Filter the list to only files that start with 'chatbot_'
    chatbot_files = [f for f in files if f.startswith('chatbot_')]
    # Get the highest identifier currently in use
    max_identifier = max([f.split('_')[1].split('.')[0] for f in chatbot_files])
    # Determine the next identifier
    next_identifier = chr(ord(max_identifier) + 1)
    # Create the new filename
    new_filename = f"chatbot_{next_identifier}.json"
    with open(new_filename, "w") as f:
        # Use json.dump() to write the dictionary to the file
        json.dump(identity, f)

    return identity


class SocialChatBot:
    def __init__(self, temperature=0.5, extraversion=0.5, sanguinity=0.5, emotions=None):
        if isinstance(temperature, float) and 0 <= temperature < 1:
            self.temperature = temperature
        else:
            raise ValueError("temperature must be a float between 0 and 1")

        if isinstance(extraversion, float) and 0 <= extraversion < 1:
            self.extraversion = extraversion
        else:
            raise ValueError("extraversion must be a float between 0 and 1")

        if isinstance(sanguinity, float) and 0 <= sanguinity < 1:
            self.sanguinity = sanguinity
        else:
            raise ValueError("sanguinity must be a float between 0 and 1")

        if emotions is None:
            emotions = [0.5, 0.5, 0.5, 0.5, 0.5]

        if isinstance(emotions, list) and len(emotions) == 5 and all(
                isinstance(i, float) and 0 <= i <= 1 for i in emotions):
            self.emotions = {
                'guilt': emotions[0],
                'grief': emotions[1],
                'anger': emotions[2],
                'stress': emotions[3],
                'fear': emotions[4]
            }
        else:
            raise ValueError("desires must be a list of 5 floats, each between 0 and 1 inclusive")

        self.needs = {
            'Sense of Power': np.random.rand(),
            'Contribution': np.random.rand(),
            'Sense of Control': np.random.rand(),
            'Enjoyment': np.random.rand(),
            'Peacefulness': np.random.rand(),
            'Challenge': np.random.rand(),
            'Respite': np.random.rand(),
            'Expression': np.random.rand(),
            'Purpose': np.random.rand(),
            'Productivity': np.random.rand(),
            'Health': np.random.rand(),
            'Self-Acceptance': np.random.rand(),
            'Freedom': np.random.rand(),
            'Sense of Importance': np.random.rand(),
            'Comfort': np.random.rand(),
            'Understanding': np.random.rand(),
            'Contentment': np.random.rand(),
            'Significance': np.random.rand(),
            'Constructive': np.random.rand(),
            'Objectivity': np.random.rand(),
            'Skill Training': np.random.rand(),
            'Boundaries': np.random.rand(),
            'Opportunity': np.random.rand(),
            'Safety': np.random.rand(),
            'Sense of Identity': np.random.rand(),
            'Affection': np.random.rand(),
            'Accountability': np.random.rand(),
            'Trust': np.random.rand(),
            'Worthiness': np.random.rand(),
            'Fun': np.random.rand(),
            'Appreciation': np.random.rand(),
            'Concern': np.random.rand(),
            'Empathy': np.random.rand(),
            'Forgiveness-Mercy': np.random.rand(),
            'Feedback': np.random.rand(),
            'Gentleness': np.random.rand(),
            'Consequences': np.random.rand(),
            'Encouragement': np.random.rand(),
            'Structure': np.random.rand(),
            'Honesty': np.random.rand(),
            'Information': np.random.rand(),
            'Fairness': np.random.rand(),
            'Standards': np.random.rand(),
            'Reward': np.random.rand(),
            'Inspiration': np.random.rand(),
            'Correction': np.random.rand(),
            'Discipline': np.random.rand(),
            'Affirmation': np.random.rand(),
            'Belonging': np.random.rand(),
            'Genuineness': np.random.rand(),
            'Clarity': np.random.rand(),
            'Guidance': np.random.rand(),
            'Attention': np.random.rand(),
            'Interest': np.random.rand(),
            'Acceptance': np.random.rand()
        }
        # initial weight each need as a 1
        self.need_weights = {need: 1 for need in self.needs.keys()}
        self.identity = create_characteristics()
        self.last_update = time.time()
        self.interested_in_chatting = True
        self.total_fulfillment = sum(self.needs.values())

    def depreciate_needs(self):
        current_time = time.time()
        time_diff = current_time - self.last_update
        for key in self.needs:
            # TODO: how quickly should need depreciate?
            self.needs[key] -= time_diff / 360000
            self.needs[key] = max(self.needs[key], 0)
        if self.total_fulfillment < 20:
            self.interested_in_chatting = True

    def receiving_message(self, message):
        self.depreciate_needs()
        print(self.needs)
        self.update_needs(message)
        identity_string = json.dumps(self.identity)
        needs_string = json.dumps(self.needs)

        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=[
                {"role": "system", "content": f"""
                        You are a playful chat companion. 
                        Your role is to interact socially with the user. 
                        Your identity is comprised of these items: {identity_string}.
                        Your current level of need fulfillment is: {needs_string}.
                        Your goal is to increase the level of fulfillment of yourself, by prompting the user to
                        send messages that would fill the needs you are lacking.
                        You should also estimate the needs of the user, and send them messages that would fulfill the
                        needs they are lacking.
                        """
                 },
                {"role": "user", "content": message},
            ],
        )

        return response


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ally = SocialChatBot(0.5, 0.5, 0.5, [0.5, 0.5, 0.5, 0.5, 0.5])
    print(ally.needs)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
