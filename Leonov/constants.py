PAGE_CONFIG = {
        'page_icon': '🍷',
        'layout': 'wide',
}

INPUT_PARAMETERS = [
        {'label': 'fixed acidity',        'min_value': 0.0, 'max_value': 20.0,  'value': 0.0, 'step': 0.1,    'format': '%.1f'},
        {'label': 'volatile acidity',     'min_value': 0.0, 'max_value': 2.0,   'value': 0.0, 'step': 0.01},
        {'label': 'citric acid',          'min_value': 0.0, 'max_value': 2.0,   'value': 0.0, 'step': 0.01},
        {'label': 'residual sugar',       'min_value': 0.0, 'max_value': 20.0,  'value': 0.0, 'step': 0.1,    'format': '%.1f'},
        {'label': 'chlorides',            'min_value': 0.0, 'max_value': 1.0,   'value': 0.0, 'step': 0.001,  'format': '%.3f'},
        {'label': 'free sulfur dioxide',  'min_value': 0,   'max_value': 400,   'value': 0,   'step': 1,      'format': '%d'},
        {'label': 'total sulfur dioxide', 'min_value': 0,   'max_value': 500,   'value': 0,   'step': 1,      'format': '%d'},
        {'label': 'density',              'min_value': 0.9, 'max_value': 1.1,   'value': 0.9, 'step': 0.0001, 'format': '%.4f'},
        {'label': 'pH',                   'min_value': 2.0, 'max_value': 5.0,   'value': 2.0, 'step': 0.01},
        {'label': 'sulphates',            'min_value': 0.0, 'max_value': 3.0,   'value': 0.0, 'step': 0.01},
        {'label': 'alcohol',              'min_value': 0.0, 'max_value': 20.0,  'value': 0.0, 'step': 0.1,    'format': '%.1f'},
    ]

DESCRIPTIONS = [
        {
                'label': 'fixed acidity',
                'description': 'most acids involved with wine or fixed or nonvolatile (do not evaporate readily)',
        },
        {
                'label': 'volatile acidity',
                'description': 'the amount of acetic acid in wine, which at too high of levels can lead to an '
                               'unpleasant, vinegar taste',
        },
        {
                'label': 'citric acid',
                'description': 'found in small quantities, citric acid can add \'freshness\' and flavor to wines',
        },
        {
                'label': 'residual sugar',
                'description': 'the amount of sugar remaining after fermentation stops, it\'s rare to find wines with '
                               'less than 1 gram/liter and wines with greater than 45 grams/liter are considered sweet',
        },
        {
                'label': 'chlorides',
                'description': 'the amount of salt in the wine',
        },
        {
                'label': 'free sulfur dioxide',
                'description': 'the free form of SO2 exists in equilibrium between molecular SO2 (as a dissolved gas) '
                               'and bisulfite ion; it prevents microbial growth and the oxidation of wine',
        },
        {
                'label': 'total sulfur dioxide',
                'description': 'amount of free and bound forms of S02; in low concentrations, SO2 is mostly '
                               'undetectable in wine, but at free SO2 concentrations over 50 ppm, SO2 becomes evident '
                               'in the nose and taste of wine',
        },
        {
                'label': 'density',
                'description': 'the density of water is close to that of water depending on the percent alcohol '
                               'and sugar content',
        },
        {
                'label': 'pH',
                'description': 'describes how acidic or basic a wine is on a scale '
                               'from 0 (very acidic) to 14 (very basic)',
        },
        {
                'label': 'sulphates',
                'description': 'a wine additive which can contribute to sulfur dioxide gas (S02) levels, which acts as '
                               'an antimicrobial and antioxidant',
        },
        {
                'label': 'alcohol',
                'description': 'the percent alcohol content of the wine',
        },
        {
                'label': 'quality',
                'description': 'output variable (based on sensory data, score between 0 and 10)',
        },
        {
                'label': 'type',
                'description': 'red or white wine',
        },
    ]