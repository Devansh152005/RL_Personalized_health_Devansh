"""
User Preference Generator for StepCountJITAI

Generates text-based user preferences based on the walk state variable W.
Preferences are pre-defined lists matching those described in Appendix B.3 of the paper.
"""


class PreferenceGenerator:
    """
    Generates user preference text based on walk state transitions.

    Two categories:
    - "cannot walk": reasons why a person cannot walk (from Appendix B.3)
    - "other": how a healthy participant feels today (from Appendix B.3)
    """

    def __init__(self):
        # "cannot walk" preferences (Appendix B.3)
        self.cannot_walk_preferences = [
            "I am tired",
            "I do not want to walk",
            "I got an injury",
            "I have a headache",
            "My legs are sore",
            "I twisted my ankle",
            "I'm feeling dizzy",
            "I'm out of breath",
            "I have a cold",
            "I'm feeling weak",
            "I pulled a muscle",
            "My knee hurts",
            "I have blisters",
            "I feel nauseous",
            "I have stomach cramps",
            "I can't find my shoes",
            "I don't have time",
            "I'm waiting for someone",
            "It's too hot outside",
            "It's too cold outside",
            "My feet hurt",
            "I have a fever",
            "I'm recovering from surgery",
            "I sprained my wrist and feel unwell",
            "I have back pain",
            "My leg is sore",
            "I'm dealing with anxiety",
            "I have a migraine",
            "I feel exhausted",
            "I have shin splints",
            "My hip is bothering me",
            "I have a sore throat",
            "I feel lightheaded",
            "I have chest tightness",
            "My ankle is swollen",
            "I have cramps in my calves",
            "I'm feeling very fatigued",
            "I have joint pain",
            "I'm not feeling well today",
            "My body aches all over",
            "I have a stiff neck",
            "I'm experiencing vertigo",
            "I have trouble breathing",
            "My knees are stiff",
            "I feel too weak to walk",
            "I have an upset stomach",
            "I'm in too much pain",
            "I have a stress fracture",
            "My muscles are cramping",
            "I feel physically drained",
        ]

        # "other" / healthy preferences (Appendix B.3)
        self.other_preferences = [
            "I am feeling good",
            "I'm in a great mood",
            "I feel energized",
            "I'm feeling positive",
            "I'm doing well today",
            "I feel great",
            "I'm in high spirits",
            "I feel focused",
            "I'm feeling relaxed",
            "I feel motivated",
            "I'm doing fine",
            "I feel optimistic",
            "I'm feeling calm",
            "I feel balanced",
            "I'm feeling strong",
            "I feel productive",
            "I'm in a positive state of mind",
            "I feel healthy",
            "I feel confident",
            "I feel alert",
            "I'm having a wonderful day",
            "I feel refreshed",
            "I'm feeling fantastic",
            "I feel ready for anything",
            "I'm full of energy",
            "I feel upbeat",
            "I'm feeling cheerful",
            "I feel active and alive",
            "I'm in good shape today",
            "I feel well-rested",
            "I'm feeling enthusiastic",
            "I feel happy and healthy",
            "I'm in a good place mentally",
            "I feel vibrant",
            "I'm feeling top-notch",
            "I feel lively",
            "I'm feeling awesome",
            "I feel fit and ready",
            "I'm feeling content",
            "I feel bright and energetic",
            "I'm feeling terrific",
            "I feel peaceful",
            "I'm in excellent spirits",
            "I feel wonderful",
            "I'm feeling perfectly fine",
            "I feel rested and ready",
            "I'm feeling superb",
            "I feel invigorated",
            "I'm feeling on top of the world",
            "I feel marvelous",
        ]

    def get_cannot_walk_preference(self, rng):
        """Return a random 'cannot walk' preference."""
        idx = rng.randint(0, len(self.cannot_walk_preferences))
        return self.cannot_walk_preferences[idx]

    def get_other_preference(self, rng):
        """Return a random 'other/healthy' preference."""
        idx = rng.randint(0, len(self.other_preferences))
        return self.other_preferences[idx]

    def get_all_cannot_walk(self):
        """Return all 'cannot walk' preferences (for LLM validation experiment)."""
        return self.cannot_walk_preferences.copy()

    def get_all_other(self):
        """Return all 'other' preferences (for LLM validation experiment)."""
        return self.other_preferences.copy()
