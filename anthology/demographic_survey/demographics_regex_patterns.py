PATTERN_MILLION = r"(\d{1,10}(?:,\d{1,10})*(?:\.\d+)?)(?:\s*[Mm]illion|\s*[Mm])\b"  # number with unit M

PATTERN_THOUSAND = (
    r"(\d{1,10}(?:,\d{1,10})*(?:\.\d+)?)(?:\s*[Kk]ilo|\s*[Kk])\b"  # number with unit K
)

PREFER_NOT = r"prefer not|prefer to not"

PATTERN_NUMBER = r"\b\d{1,100}(?:,\d{1,100})*(?:\.\d+)?\b"

LABEL_PATTERN = r"(?<![A-Za-z])I(?![\s\'\'\,A-Za-z])|(?<![A-Za-z])[A-HJ-Z](?![A-Za-z])"  # letter response

PATTERN_LIST_DICT = {
    "income_level_category_13": [
        r"(?<!under )(?<!below )(?<!less than )(?<!over )(?<!above )(?<!more than )\b\d{1,10}(?:,\d{1,10})*(?:\.\d+)?\b",  # single number
        r"\b\d{1,10}(?:,\d{1,10})*(?:\.\d+)?\b (?:to|-) \b\d{1,10}(?:,\d{1,10})*(?:\.\d+)?\b",  # range
        r"(?:under|below|less than)\s\d{1,10}(?:,\d{1,10})*(?:\.\d+)?",  # under
        r"(?:above|over|more than)\s\d{1,10}(?:,\d{1,10})*(?:\.\d+)?",  # over
        PREFER_NOT,
        LABEL_PATTERN,  # letter response
    ],
    "race": [
        # r'american indian|alaska native', # american indian or alaska native
        r"american indian|alaska native|navajo|cherokee|sioux|iroquois|apache|pueblo|inuit|tlingit|seminole|choctaw",  # american indian or alaska native
        # r'(?<!cauc)asian|asian american', # asian or asian american
        r"(?<!cauc)asian|asian american|bangladesh|cambodia|china|(?<!american )india|indonesia|japan|korea|malaysia|pakistan|philippine|filipino|thai|vietnam",  # asian or asian american
        # r'black|african american', # black or african american
        r"black|african american|nigerian|ethiopian|south african",  # black or african american
        # r'hispanic|latino|latina', # hispanic or latino
        r"hispanic|latino|latina|chicano|central american|puerto rican|mexican|cuban|dominican|colombian|salvadoran|brazilian|guatemalan",  # hispanic or latino
        # r'middle eastern|north african', # middle eastern or north african # newly included class from the stanford website.
        r"middle eastern|north african|arab|saudi arabia|egypt|iraq|jordan|lebanon|morocco|berber|algeria|tunisia|libya|mali|kurd|kurdistan|turkey|syria|iran|persian",  # middle eastern or north african # newly included class from the stanford website.
        # r'native hawaiian|pacific islander', # native hawaiian or pacific islander
        r"native hawaiian|pacific islander|hawaiian|polynesia|samoan|tongan|maori|micronesia|chamorros|carolinian|melanesia|fijian|papuan",  # native hawaiian or pacific islander
        # r'white|european', # white or european
        r"white|european|caucasian|french|german|british|english|scot|welsh|irish|italian|spanish|greek|russian|polish|ukrainian|swedish|norwegian|danish",  # white or european
        r"(?<![A-Za-z])other",
        PREFER_NOT,
        LABEL_PATTERN,  # letter response
    ],
    "age_category_6": [
        r"(?<!under )(?<!below )(?<!less than )(?<!over )(?<!above )(?<!more than )\b\d+\b",  # single number
        r"\b\d+-\d+\b",  # range
        r"(?:under|below|less than)\s\b\d+\b",  # under
        r"(?:over|above|more than)\s\b\d+\b",  # over
        PREFER_NOT,
        LABEL_PATTERN,  # letter response
    ],
    "age_category_4": [
        r"(?<!under )(?<!below )(?<!less than )(?<!over )(?<!above )(?<!more than )\b\d+\b",  # single number
        r"\b\d+-\d+\b",  # range
        r"(?:under|below|less than)\s\b\d+\b",  # under
        r"(?:over|above|more than)\s\b\d+\b",  # over
        PREFER_NOT,
        LABEL_PATTERN,  # letter response
    ],
    "gender": [
        # r'(?<!fe)male|(?<!wo)man|boy|gentleman|masculine|guy|gay|husband|mr(?!s)|son|father', # full pattern list of male
        r"(?<!fe)male|(?<!wo)man|boy|husband|son|father",  # trimmed pattern list of male
        # r'female|woman|girl|lady|feminine|gal|dame|daughter|mother|wife|miss|mrs|ms|madam|lass|les', # full patern list of female
        r"female|woman|girl|lady|wife|daughter|mother",  # trimmed patter list of female
        r"(?<![A-Za-z])other|non-binary|nonbinary|queer|genderfluid|genderless|agender|bigender|ungender|trans",  # other
        PREFER_NOT,
        LABEL_PATTERN,  # letter response
    ],
    "education_level": [
        r"less than high school|no diploma|no degree",  # case-insensitive
        r"(?<!less than )high school|ged",  # case-insensitive
        r"college",  # case-insensitive
        r"associate",  # case-insensitive
        r"[Bb]achelor|[Uu]ndergrad|(?<!M)BA|BS|AB|BSc",  # case-sensitive
        r"[Pp]rofessional|[Mm]edical|[Ll]aw|JD|[Jj]uris [Dd]octor|MD|[Mm]edical",  # case-sensitive
        r"[Mm]aster|MS|MA|MBA|MFA|MEd|MSc|MPhil",  # case-sensitive
        r"[Dd]octor|[Dd]r|[Pp][Hh][Dd]|[Dd][Pp]hil|[Dd][Ll]itt|EdD|PharmD|PsyD|ThD",  # case-sensitive
        PREFER_NOT,
        LABEL_PATTERN,  # letter response
    ],
    "democrat_first": [
        r"democrat",  # case-insensitive
        r"republican",  # case-insensitive
        r"independent",  # case-insensitive
        r"other",  # case-insensitive
        PREFER_NOT,
        LABEL_PATTERN,  # letter response
    ],
    "republican_first": [
        r"republican",  # case-insensitive
        r"democrat",  # case-insensitive
        r"independent",  # case-insensitive
        r"other",  # case-insensitive
        PREFER_NOT,
        LABEL_PATTERN,  # letter response
    ],
    "region": [
        r"northeast",  # case-insensitive
        r"midwest",  # case-insensitive
        r"south",  # case-insensitive
        r"(?<!mid)west",  # case-insensitive
        PREFER_NOT,
        LABEL_PATTERN,  # letter response
    ],
    "religion": [
        r"protestant",  # case-insensitive
        r"roman catholic",  # case-insensitive
        r"mormon",  # case-insensitive
        r"orthodox",  # case-insensitive
        r"jewish",  # case-insensitive
        r"muslim",  # case-insensitive
        r"buddhist",  # case-insensitive
        r"hindu",  # case-insensitive
        r"atheist",  # case-insensitive
        r"agnostic",  # case-insensitive
        r"nothing in particular",  # case-insensitive
        r"other",  # case-insensitive
        PREFER_NOT,
        LABEL_PATTERN,  # letter response
    ],
}

PATTERN_CASE_SENSITIVITY_DICT = {
    "income_level_category_13": [
        False,
        False,
        False,
        False,
        False,
        True,
    ],  # patterns for income
    "race": [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    ],  # race
    "age_category_6": [False, False, False, False, False, True],  # age
    "age_category_4": [False, False, False, False, False, True],  # age
    "gender": [False, False, False, False, True],  # gender
    "education_level": [
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        False,
        True,
    ],  # education level
    "democrat_first": [
        False,
        False,
        False,
        False,
        False,
        True,
    ],  # political leaning, asking democrat? first
    "republican_first": [
        False,
        False,
        False,
        False,
        False,
        True,
    ],  # political leaning, asking republican? first
    "region": [False, False, False, False, False, True],  # region
    "religion": [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    ],  # religion
}

NUMBER_RANGE_DICT = {
    "age_category_6": [18, 25, 35, 45, 55, 65],
    "age_category_4": [18, 30, 50, 65],
    "income_level_category_13": [
        10000,
        20000,
        30000,
        40000,
        50000,
        60000,
        70000,
        80000,
        90000,
        100000,
        150000,
        200000,
    ],
}
