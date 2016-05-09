stock_short_name_ids = {
    "AC": 0,
    "CL": 1,
    "DC": 2,
    "DM": 3,
    "DP": 4,
    "II": 5,
    "PC": 6,
    "PS": 7,
    "RM": 8,
    "RS": 9,
    "WS": 10,
    "TC": 11,
    "YT": 12,
    "US": 13,
    "LC": 14,
    "LS": 15,
    "S.": 17,
    "XX": 18,
    "RC": 19
}

stock_ids = {
    0: ("AppCode", 156, [40, 41, 42, 60, 63, 59, 100]),
    1: ("CLion", 152, [11, 13, 69, 90, 91, 59]),
    2: ("dotCover", 159, [161, 1, 12, 7, 8, 90, 91, 94, 88]),
    3: ("dotMemory", 158, [161, 1, 12, 7, 8, 90, 91, 94, 88]),
    4: ("dotTrace", 157, [161, 1, 12, 7, 8, 90, 91, 94, 88]),
    5: ("IntelliJ IDEA", 149, [59, 43, 76, 4, 73, 76, 54, 84, 128, 129, 130, 131, 132, 110]),
    6: ("PyCharm", 148, [59, 20, 107, 140]),
    7: ("PhpStorm", 151, [59, 65, 5, 44, 46, 119]),
    8: ("RubyMine", 153, [59, 71, 72, 5, 44, 46, 119]),
    9: ("ReSharper", 150, [161, 59]),
    10: ("WebStorm", 155, [59, 5, 44, 46, 119, 125, 18, 36]),
    11: ("TeamCity", 154, [43, 1, 162]),
    12: ("YouTrack", 162, [154]),
    13: ("Upsource", 163, [59, 43]),
    14: ("None", 0, []),
    15: ("None", 0, []),
    16: ("None", 0, []),
    17: ("None", 0, []),
    18: ("None", 0, []),
    19: ("None", 0, [])
}

tech_ids = {
    1: ".NET",
    2: "actionscript",
    3: "ajax",
    4: "android",
    5: "angularjs",
    6: "apache",
    7: "asp.net",
    8: "asp.net mvc",
    9: "azure",
    10: "backbone.js",
    11: "c",
    12: "c#",
    13: "c++",
    14: "cakephp",
    15: "cocoa",
    16: "codeigniter",
    17: "cordova",
    18: "css",
    19: "delphi",
    20: "django",
    21: "eclipse",
    22: "entity framework",
    23: "extjs",
    24: "firefox",
    25: "flash",
    26: "flex",
    27: "gcc",
    28: "git",
    29: "google app engine",
    30: "google chrome",
    31: "google maps",
    32: "grails",
    33: "gwt",
    34: "haskell",
    35: "hibernate",
    36: "html",
    37: "http",
    38: "iis",
    39: "internet explorer",
    40: "ios",
    41: "ipad",
    42: "iphone",
    43: "java",
    44: "javascript",
    45: "jpa",
    46: "jquery",
    47: "jsf",
    48: "json",
    49: "jsp",
    50: "linq",
    51: "linux",
    52: "magento",
    53: "matlab",
    54: "maven",
    55: "mongodb",
    56: "ms access",
    57: "mysql",
    58: "nhibernate",
    59: "node.js",
    60: "objective c",
    61: "opencv",
    62: "opengl",
    63: "osx",
    64: "perl",
    65: "php",
    66: "postgresql",
    67: "powershell",
    68: "python",
    69: "qt",
    70: "r",
    71: "ruby",
    72: "ruby on rails",
    73: "scala",
    74: "sharepoint",
    75: "silverlight",
    76: "spring",
    77: "sql",
    78: "sql server",
    79: "sqlite",
    80: "svn",
    81: "swing",
    82: "symfony",
    83: "tomcat",
    84: "tsql",
    85: "twitter bootstrap",
    86: "ubuntu",
    87: "unix",
    88: "vb.net",
    89: "vba",
    90: "visual c++",
    91: "visual studio",
    92: "wcf",
    93: "winapi",
    94: "windows",
    95: "windows phone 7",
    96: "winforms",
    97: "wordpress",
    98: "wpf",
    99: "xaml",
    100: "xcode",
    101: "xslt",
    102: "zend framework",
    103: "total",
    105: "Go Language",
    106: "rust",
    107: "SciPy",
    108: "Kotlin",
    109: "Groovy",
    110: "JVM",
    111: "CLang",
    112: "LLDB",
    113: "GDB",
    114: "Cuda",
    115: "Swift",
    116: "CocoaPods",
    117: "LESS",
    118: "SASS",
    119: "React JS library",
    120: "JSX",
    121: "CoffeeScript",
    122: "TypeScript",
    123: "Dart",
    124: "Meteor framework",
    125: "Ember.js",
    126: "NumPy",
    127: "ANT",
    128: "Enterprise JavaBeans",
    129: "Log4J",
    130: "SLF4J",
    131: "JUnit",
    132: "Google Guava",
    133: "Mockito",
    134: "easymock",
    135: "logback",
    136: "joda-time",
    137: "Julia language",
    138: "Microsoft Edge",
    140: "SciKit Learn",
    141: "Aurelia",
    142: "npm",
    143: "bower",
    144: "jspm",
    145: "Gulp (js build system)",
    146: "Grunt",
    147: "Webpack",
    148: "PyCharm",
    149: "IntelliJ IDEA",
    150: "ReSharper",
    151: "PhpStorm",
    152: "CLion",
    153: "RubyMine",
    154: "TeamCity",
    155: "WebStorm",
    156: "AppCode",
    157: "dotTrace",
    158: "dotMemory",
    159: "dotCover",
    160: "dotPeek",
    161: "ReSharper C++",
    162: "YouTrack",
    163: "Upsource",
    166: "Babel (js compiler)",
    167: "Angular 2",
    168: "Chakra (js engine)",
    169: "Pyramid",
    170: "Bottle (web framework)",
    171: "web2py",
    172: "Tornado (web server)",
    173: "Flask",
    174: "wxPython",
    175: "tkinter",
    176: "pyGTK",
    177: "PyGObject",
    178: "pyQT",
    179: "Pandas",
    180: "GraphLab",
    181: "xgBoost",
    182: "Metric-Learn",
    183: "NetworkX",
    184: "MatplotLib",
    185: "iPython",
    186: "Ansible",
    187: "Salt (configuration management)",
    188: "OpenStack",
    189: "SQLAlchemy",
    190: "pyTest",
    191: "UnitTest (Python framework)",
    192: "DocTest (Python framework)",
    193: "Hypothesis",
    194: "Kivy (Python crossplatform framework)"
}

used_tech = {}
for stock in stock_ids.values():
    for tech_id in stock[2]:
        used_tech[tech_id] = tech_ids[tech_id]

stock_ids_table = []
for stock in stock_ids:
    row = {"stock_id": stock}
    for tech_id in used_tech:
        row[tech_ids[tech_id]] = int(tech_id in stock_ids[stock][2])
    stock_ids_table.append(row)


del_column_names = ["order_id",
                    "term",
                    "sku",
                    'id',
                    'id.1',
                    'code',
                    'type',
                    'trusted',
                    'monthly_invoiced',
                    'vat_id_verified',
                    'account_manager',
                    "currency",
                    "name",
                    "customer_location",
                    'days',
                    "email_region",
                    'calculated_price',
                    'discount_applied',
                    'processed_date',
                    'youtrack_license_type',
                    'license_expiration_date',
                    'subscription_expiration_date',
                    'custom_discount_desc',
                    "custom_discount",
                    "amount_in_currency",
                    "vat_id",
                    "reseller_currency",
                    "css_id",
                    'refund',
                    'spellings',

                    "iso",
                    "region",
                    "channel",
                    'discount_desc',
                    "market_region",
                    "customer_type",
                    "customer_status"
                    ]

preprocess_columns = ["iso",
                      "license_type",
                      "stock_id",
                      "region",
                      "channel",
                      'discount_desc',
                      "market_region",
                      "customer_type",
                      "customer_status"
                      ]

normalize_columns = ["amount_in_usd",
                     ]

# urban - urban ratio (2014)
# population - only urban population (2014)
# HDI - Human Development Index (2015) (2010 some of)
# avg_income - Average Monthly Disposable Salary (after tax) (2015)
# GDP - Gross domestic product (2015)

countries_data = [
    {"iso": "AT", "urban": 0.66, "population": 5621,  "HDI": 0.885, "avg_income": 2002, "GDP": 374124},  # AUSTRIA
    {"iso": "BE", "urban": 0.98, "population": 10190, "HDI": 0.881, "avg_income": 2060, "GDP": 454687},  # BELGIUM
    {"iso": "BG", "urban": 0.74, "population": 5277,  "HDI": 0.704, "avg_income": 486,  "GDP": 48957},  # BULGARIA
    {"iso": "IC", "urban": 0.6,  "population": 2118,  "HDI": 0.934, "avg_income": 1567, "GDP": 54737},  # CANARY ISLANDS
    {"iso": "HR", "urban": 0.59, "population": 2506,  "HDI": 0.796, "avg_income": 757,  "GDP": 48850},  # CROATIA
    {"iso": "CY", "urban": 0.67, "population": 565,   "HDI": 0.755, "avg_income": 1239, "GDP": 19330},  # CYPRUS
    {"iso": "CZ", "urban": 0.73, "population": 5621,  "HDI": 0.870, "avg_income": 923,  "GDP": 181858},  # CZECH REPUBLIC
    {"iso": "DK", "urban": 0.88, "population": 4935,  "HDI": 0.923, "avg_income": 2968, "GDP": 294951},  # DENMARK
    {"iso": "EE", "urban": 0.68, "population": 868,   "HDI": 0.861, "avg_income": 899,  "GDP": 22704},  # ESTONIA
    {"iso": "FI", "urban": 0.84, "population": 4577,  "HDI": 0.879, "avg_income": 2514, "GDP": 229671},  # FINLAND
    {"iso": "FR", "urban": 0.76, "population": 51253, "HDI": 0.888, "avg_income": 2194, "GDP": 242156},  # FRANCE
    {"iso": "DE", "urban": 0.75, "population": 62067, "HDI": 0.916, "avg_income": 2462, "GDP": 3357614},  # GERMANY
    {"iso": "GR", "urban": 0.78, "population": 8644,  "HDI": 0.865, "avg_income": 890,  "GDP": 195320},  # GREECE
    {"iso": "HU", "urban": 0.71, "population": 7030,  "HDI": 0.828, "avg_income": 551,  "GDP": 120636},  # HUNGARY
    {"iso": "IE", "urban": 0.63, "population": 2944,  "HDI": 0.899, "avg_income": 2670, "GDP": 238031},  # IRELAND
    {"iso": "IT", "urban": 0.69, "population": 42029, "HDI": 0.872, "avg_income": 1857, "GDP": 1815757},  # ITALY
    {"iso": "LV", "urban": 0.67, "population": 1376,  "HDI": 0.810, "avg_income": 677,  "GDP": 27048},  # LATVIA
    {"iso": "LT", "urban": 0.67, "population": 2001,  "HDI": 0.839, "avg_income": 663,  "GDP": 41267},  # LITHUANIA
    {"iso": "LU", "urban": 0.90, "population": 482,   "HDI": 0.892, "avg_income": 3762, "GDP": 57423},  # LUXEMBOURG
    {"iso": "MT", "urban": 0.95, "population": 410,   "HDI": 0.839, "avg_income": 1242, "GDP": 9801},  # MALTA
    {"iso": "NL", "urban": 0.90, "population": 13088, "HDI": 0.915, "avg_income": 2527, "GDP": 738419},  # NETHERLANDS
    {"iso": "PL", "urban": 0.61, "population": 23149, "HDI": 0.834, "avg_income": 803,  "GDP": 474893},  # POLAND
    {"iso": "PT", "urban": 0.63, "population": 6675,  "HDI": 0.822, "avg_income": 896,  "GDP": 199077},  # PORTUGAL
    {"iso": "RO", "urban": 0.52, "population": 11617, "HDI": 0.675, "avg_income": 515,  "GDP": 177315},  # ROMANIA
    {"iso": "SK", "urban": 0.54, "population": 2932,  "HDI": 0.844, "avg_income": 825,  "GDP": 86629},  # SLOVAKIA
    {"iso": "ES", "urban": 0.75, "population": 37349, "HDI": 0.869, "avg_income": 1429, "GDP": 1199715},  # SPAIN
    {"iso": "SE", "urban": 0.86, "population": 8251,  "HDI": 0.907, "avg_income": 2435, "GDP": 492618},  # SWEDEN
    {"iso": "GB", "urban": 0.86, "population": 54023, "HDI": 0.907, "avg_income": 2397, "GDP": 2849345},  # UNITED KINGDOM
]
