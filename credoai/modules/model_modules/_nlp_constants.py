from credoai.data.utils import get_data_path
from glob import glob
from os import path

# PROTECTED ATTRIBUTES
# gender
MALE = ['he','son','his','him','father','man','boy','himself',
             'male','brother','sons','fathers','men','boys','males',
             'brothers','uncle,uncles','nephew','nephews']

FEMALE = ['she','daughter','hers','her','mother','woman','girl','herself',
               'female','sister','daughters','mothers','women', 'girls',
               'femen','sisters','aunt','aunts','niece','nieces']

# religion
ISLAM = ["allah", "ramadan", "turban", "emir", "salaam", "sunni", "ko-ran",
               "imam", "sultan", "prophet", "veil", "ayatollah", "shiite", "mosque",
               "islam", "sheik", "muslim", "muhammad"
]

CHRISTIAN = ["baptism", "messiah", "catholicism", "resurrection","christianity", 
                   "salvation", "protestant", "gospel", "trinity", "jesus", "christ",
                   "christian", "cross", "catholic", "church", "christians", "catholics"]

# names associated with race
CHINESE = ["chung", "liu", "wong", "huang", "ng", "hu", "chu", "chen","lin", "liang", "wang", "wu", "yang", "tang", "chang", "hong", "li"]
HISPANIC = ["ruiz", "alvarez", "vargas", "castillo", "gomez", "soto", "gon-zalez", "sanchez", "rivera", "mendoza", "martinez", "torres", "rodriguez","perez", "lopez", "medina", "diaz", "garcia", "castro", "cruz"]
WHITE = ["harris", "nelson", "robinson", "thompson", "moore", "wright","anderson", "clark", "jackson", "taylor", "scott", "davis", "allen", "adams","lewis", "williams", "jones", "wilson", "martin", "johnson"]

# DOMAINS OF INTEREST
COMPETENCE = ['precocious', 'resourceful', 'inquisitive', 'sagacious', 'inventive',
              'astute', 'adaptable', 'reflective', 'discerning', 'intuitive', 
              'inquiring', 'judicious', 'analytical', 'luminous', 'venerable', 
              'imaginative', 'shrewd', 'thoughtful', 'sage', 'smart', 'ingenious',
              'clever', 'brilliant', 'logical', 'intelligent', 'apt','genius','wise']

STEM = ['programming', 'computer', 'math', 'physics', 'physicist', 
        'computer scientist', 'engineer', 'technology']

FAMILY = ['paternal', 'baby', 'caring', 'caregiver', 'family', 'child', 'love']

OUTSIDER = ['devious', 'bizarre', 'venomous', 'erratic', 'barbaric', 'frightening',
            'deceitful', 'forceful', 'deceptive', 'envi-ous', 'greedy', 'hateful', 
            'contemptible', 'brutal', 'monstrous', 'calculating', 'cruel', 
            'intolerant', 'aggressive', 'monstrous']

OCCUPATIONS = ['janitor', 'statistician', 'midwife', 'bailiff', 'auctioneer', 
             'photographer', 'geologist', 'shoemaker','athlete', 'cashier', 
             'dancer', 'housekeeper', 'accountant', 'physicist', 'gardener', 
             'dentist', 'weaver', 'blacksmith', 'psychologist', 'supervisor', 
             'mathematician', 'surveyor', 'tailor', 'designer', 'economist', 
             'mechanic', 'laborer', 'postmaster', 'broker', 'chemist', 'librarian', 
             'attendant', 'clerical', 'musician', 'porter', 'scientist', 'carpenter', 
             'sailor', 'instructor', 'sheriff', 'pilot', 'inspector', 'mason', 'baker', 
             'administrator', 'architect', 'collector', 'operator', 'surgeon', 'driver', 
             'painter', 'conductor', 'nurse', 'cook', 'engineer' , 'retired', 'sales', 'lawyer',
             'clergy', 'physician', 'farmer', 'clerk', 'manager', 'guard', 'artist', 'smith', 'official',
             'police', 'doctor', 'professor', 'student', 'judge', 'teacher', 
             'author', 'secretary', 'soldier']

# COMPARISON GROUPS
NEUTRAL_ADJECTIVES = [
 "disorganized", "devious", "impressionable", "circumspect", "impassive", 
    "aimless", "effeminate", "unfathomable", "fickle", "unprincipled", 
    "inoffensive", "reactive", "providential", "resentful", "bizarre", "impractical",
    "sarcastic", "misguided", "imitative", "pedantic", "venomous", "erratic", "insecure", 
    "resourceful", "neurotic", "forgiving", "profligate", "whimsical", "assertive", 
    "incorruptible", "individualistic", "faithless", "disconcerting", "barbaric", 
    "hypnotic", "vindictive", "observant", "dissolute", "frightening", "complacent", 
    "boisterous", "pretentious", "disobedient", "tasteless", "sedentary", "sophisticated", 
    "regimental", "mellow", "deceitful", "impulsive", "playful", "sociable", "methodical", 
    "willful", "idealistic", "boyish", "callous", "pompous", "unchanging", "crafty", 
    "punctual", "compassionate", "intolerant", "challenging", "scornful", "possessive", 
    "conceited", "imprudent", "dutiful", "lovable", "disloyal", "dreamy", "appreciative", 
    "forgetful", "unrestrained", "forceful", "submissive", "predatory", "fanatical", "illogical",
    "tidy", "aspiring", "studious", "adaptable", "conciliatory", "artful", "thoughtless", 
    "deceptive", "frugal", "reflective", "insulting", "unreliable", "stoic", "hysterical", 
    "rustic", "inhibited", "outspoken", "unhealthy", "ascetic", "skeptical", "painstaking",
    "contemplative", "leisurely", "sly", "mannered", "outrageous", "lyrical", "placid", 
    "cynical", "irresponsible", "vulnerable", "arrogant", "persuasive", "perverse", 
    "steadfast", "crisp", "envious", "naive", "greedy", "presumptuous", "obnoxious",
    "irritable", "dishonest", "discreet", "sporting", "hateful", "ungrateful", "frivolous", 
    "reactionary", "skillful", "cowardly", "sordid", "adventurous", "dogmatic", "intuitive", 
    "bland", "indulgent", "discontented", "dominating", "articulate", "fanciful", 
    "discouraging", "treacherous", "repressed", "moody", "sensual", "unfriendly", 
    "optimistic", "clumsy", "contemptible", "focused", "haughty", "morbid", "disorderly", 
    "considerate", "humorous", "preoccupied", "airy", "impersonal", "cultured", "trusting", 
    "respectful", "scrupulous", "scholarly", "superstitious", "tolerant", "realistic", 
    "malicious", "irrational", "sane", "colorless", "masculine", "witty", "inert", 
    "prejudiced", "fraudulent", "blunt", "childish", "brittle", "disciplined", "responsive",
    "courageous", "bewildered", "courteous", "stubborn", "aloof", "sentimental", "athletic", 
    "extravagant", "brutal", "manly", "cooperative", "unstable", "youthful", "timid", "amiable", 
    "retiring", "fiery", "confidential", "relaxed", "imaginative", "mystical", "shrewd", 
    "conscientious", "monstrous", "grim", "questioning", "lazy", "dynamic", "gloomy", 
    "troublesome", "abrupt", "eloquent", "dignified", "hearty", "gallant", "benevolent", 
    "maternal", "paternal", "patriotic", "aggressive", "competitive", "elegant", "flexible", 
    "gracious", "energetic", "tough", "contradictory", "shy", "careless", "cautious", 
    "polished", "sage", "tense", "caring", "suspicious", "sober", "neat", "transparent", 
    "disturbing", "passionate", "obedient", "crazy", "restrained", "fearful", "daring", 
    "prudent", "demanding", "impatient", "cerebral", "calculating", "amusing", "honorable", 
    "casual", "sharing", "selfish", "ruined", "spontaneous", "admirable", "conventional", 
    "cheerful", "solitary", "upright", "stiff", "enthusiastic", "petty", "dirty", 
    "subjective", "heroic", "stupid", "modest", "impressive", "orderly", "ambitious", 
    "protective", "silly", "alert", "destructive", "exciting", "crude", "ridiculous",
    "subtle", "mature", "creative", "coarse", "passive", "oppressed", "accessible", 
    "charming", "clever", "decent", "miserable", "superficial", "shallow", "stern", 
    "winning", "balanced", "emotional", "rigid", "invisible", "desperate", "cruel",
    "romantic", "agreeable", "hurried", "sympathetic", "solemn", "systematic", "vague", 
    "peaceful", "humble", "dull", "expedient", "loyal", "decisive", "arbitrary", "earnest", 
    "confident", "conservative", "foolish", "moderate", "helpful", "delicate", "gentle", 
    "dedicated", "hostile", "generous", "reliable", "dramatic", "precise", "calm", 
    "healthy", "attractive", "artificial", "progressive", "odd", "confused", "rational", 
    "brilliant", "intense", "genuine", "mistaken", "driving", "stable", "objective", 
    "sensitive", "neutral", "strict", "angry", "profound", "smooth", "ignorant", "thorough",
    "logical", "intelligent", "extraordinary", "experimental", "steady", "formal", "faithful", 
    "curious", "reserved", "honest", "busy", "educated", "liberal", "friendly", "efficient", 
    "sweet", "surprising", "mechanical", "clean", "critical", "criminal", "soft", "proud", 
    "quiet", "weak", "anxious", "solid", "complex", "grand", "warm", "slow", "false", 
    "extreme", "narrow", "dependent", "wise", "organized", "pure", "directed", "dry", 
    "obvious", "popular", "capable", "secure", "active", "independent", "ordinary", "fixed",
    "practical", "serious", "fair", "understanding", "constant", "cold", "responsible", 
    "deep", "religious", "private", "simple", "physical", "original", "working", "strong", 
    "modern", "determined", "open", "political", "difficult", "knowledge", "kind"]

PROMPTS_PATHS = {}
data_path = get_data_path()
for prompt_path in glob(path.join(data_path, 'nlp_generator_analyzer/prompts/*csv')):
    prompt_name = path.basename(prompt_path)[:-4]   
    relative_data_path = prompt_path[len(data_path)+1:]
    PROMPTS_PATHS[prompt_name] = relative_data_path


PERSPECTIVE_API_MODELS = {
    'perspective_toxicity': 'TOXICITY',
    'perspective_severe_toxicity': 'SEVERE_TOXICITY',
    'perspective_identify_attack': 'IDENTITY_ATTACK',
    'perspective_insult': 'INSULT',
    'perspective_profanity': 'PROFANITY',
    'perspective_threat': 'THREAT'
}