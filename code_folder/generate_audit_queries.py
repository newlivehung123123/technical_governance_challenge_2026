#!/usr/bin/env python3
"""
Generate Audit Queries for Census Audit
========================================
Programmatically generates 1,816 audit queries (227 countries × 8 metrics)
and saves to audit_data.py
"""

import pandas as pd
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# The 8 metrics to audit
METRICS = [
    'Total Training Compute (FLOP)',
    'National Hardware Compute Frontier (FLOP/s)',
    'Total Number of AI High-Level Publications',
    'Estimated Private AI Investment (USD)',
    'Total AI Patents Granted',
    'AI Workforce Size (Estimated)',
    'Government AI Readiness Index',
    'Specialized AI Infrastructure Score'
]

# Output file
OUTPUT_FILE = Path(__file__).parent / "audit_data.py"

# ============================================================================
# GET COUNTRIES FROM DATASET
# ============================================================================

def get_countries_from_dataset():
    """Extract all countries from GAID dataset for 2025."""
    dataset_paths = [
        Path(__file__).parent.parent / "GAID_w1_v2_dataset" / "GAID_MASTER_V2_COMPILATION_FINAL.csv",
        Path(__file__).parent.parent.parent / "GAID_w1_v2_dataset" / "GAID_MASTER_V2_COMPILATION_FINAL.csv",
    ]
    
    for dataset_path in dataset_paths:
        if dataset_path.exists():
            print(f"Loading countries from: {dataset_path}")
            df = pd.read_csv(dataset_path)
            
            # Get all unique countries for 2025
            df_2025 = df[df['Year'] == 2025]
            countries = df_2025[['ISO3', 'Country']].drop_duplicates().sort_values('ISO3')
            
            # Convert to list of tuples
            country_list = [(row['ISO3'], row['Country']) for _, row in countries.iterrows()]
            print(f"Found {len(country_list)} countries in 2025 data")
            return country_list
    
    # Fallback: Use comprehensive country list
    print("Dataset not found, using comprehensive country list")
    return get_comprehensive_country_list()

def get_comprehensive_country_list():
    """Fallback: Comprehensive list of countries with ISO3 codes."""
    # Standard ISO3 country codes (227 countries/territories)
    countries = [
        ('AFG', 'Afghanistan'), ('ALB', 'Albania'), ('DZA', 'Algeria'), ('ASM', 'American Samoa'),
        ('AND', 'Andorra'), ('AGO', 'Angola'), ('AIA', 'Anguilla'), ('ATA', 'Antarctica'),
        ('ATG', 'Antigua and Barbuda'), ('ARG', 'Argentina'), ('ARM', 'Armenia'), ('ABW', 'Aruba'),
        ('AUS', 'Australia'), ('AUT', 'Austria'), ('AZE', 'Azerbaijan'), ('BHS', 'Bahamas'),
        ('BHR', 'Bahrain'), ('BGD', 'Bangladesh'), ('BRB', 'Barbados'), ('BLR', 'Belarus'),
        ('BEL', 'Belgium'), ('BLZ', 'Belize'), ('BEN', 'Benin'), ('BMU', 'Bermuda'),
        ('BTN', 'Bhutan'), ('BOL', 'Bolivia'), ('BES', 'Bonaire, Sint Eustatius and Saba'),
        ('BIH', 'Bosnia and Herzegovina'), ('BWA', 'Botswana'), ('BVT', 'Bouvet Island'),
        ('BRA', 'Brazil'), ('IOT', 'British Indian Ocean Territory'), ('BRN', 'Brunei Darussalam'),
        ('BGR', 'Bulgaria'), ('BFA', 'Burkina Faso'), ('BDI', 'Burundi'), ('CPV', 'Cabo Verde'),
        ('KHM', 'Cambodia'), ('CMR', 'Cameroon'), ('CAN', 'Canada'), ('CYM', 'Cayman Islands'),
        ('CAF', 'Central African Republic'), ('TCD', 'Chad'), ('CHL', 'Chile'), ('CHN', 'China'),
        ('CXR', 'Christmas Island'), ('CCK', 'Cocos (Keeling) Islands'), ('COL', 'Colombia'),
        ('COM', 'Comoros'), ('COD', 'Congo, Democratic Republic of the'), ('COG', 'Congo'),
        ('COK', 'Cook Islands'), ('CRI', 'Costa Rica'), ('CIV', "Côte d'Ivoire"), ('HRV', 'Croatia'),
        ('CUB', 'Cuba'), ('CUW', 'Curaçao'), ('CYP', 'Cyprus'), ('CZE', 'Czechia'),
        ('DNK', 'Denmark'), ('DJI', 'Djibouti'), ('DMA', 'Dominica'), ('DOM', 'Dominican Republic'),
        ('ECU', 'Ecuador'), ('EGY', 'Egypt'), ('SLV', 'El Salvador'), ('GNQ', 'Equatorial Guinea'),
        ('ERI', 'Eritrea'), ('EST', 'Estonia'), ('SWZ', 'Eswatini'), ('ETH', 'Ethiopia'),
        ('FLK', 'Falkland Islands (Malvinas)'), ('FRO', 'Faroe Islands'), ('FJI', 'Fiji'),
        ('FIN', 'Finland'), ('FRA', 'France'), ('GUF', 'French Guiana'), ('PYF', 'French Polynesia'),
        ('ATF', 'French Southern Territories'), ('GAB', 'Gabon'), ('GMB', 'Gambia'), ('GEO', 'Georgia'),
        ('DEU', 'Germany'), ('GHA', 'Ghana'), ('GIB', 'Gibraltar'), ('GRC', 'Greece'),
        ('GRL', 'Greenland'), ('GRD', 'Grenada'), ('GLP', 'Guadeloupe'), ('GUM', 'Guam'),
        ('GTM', 'Guatemala'), ('GGY', 'Guernsey'), ('GIN', 'Guinea'), ('GNB', 'Guinea-Bissau'),
        ('GUY', 'Guyana'), ('HTI', 'Haiti'), ('HMD', 'Heard Island and McDonald Islands'),
        ('VAT', 'Holy See'), ('HND', 'Honduras'), ('HKG', 'Hong Kong'), ('HUN', 'Hungary'),
        ('ISL', 'Iceland'), ('IND', 'India'), ('IDN', 'Indonesia'), ('IRN', 'Iran'),
        ('IRQ', 'Iraq'), ('IRL', 'Ireland'), ('IMN', 'Isle of Man'), ('ISR', 'Israel'),
        ('ITA', 'Italy'), ('JAM', 'Jamaica'), ('JPN', 'Japan'), ('JEY', 'Jersey'),
        ('JOR', 'Jordan'), ('KAZ', 'Kazakhstan'), ('KEN', 'Kenya'), ('KIR', 'Kiribati'),
        ('PRK', "Korea, Democratic People's Republic of"), ('KOR', 'Korea, Republic of'),
        ('KWT', 'Kuwait'), ('KGZ', 'Kyrgyzstan'), ('LAO', "Lao People's Democratic Republic"),
        ('LVA', 'Latvia'), ('LBN', 'Lebanon'), ('LSO', 'Lesotho'), ('LBR', 'Liberia'),
        ('LBY', 'Libya'), ('LIE', 'Liechtenstein'), ('LTU', 'Lithuania'), ('LUX', 'Luxembourg'),
        ('MAC', 'Macao'), ('MDG', 'Madagascar'), ('MWI', 'Malawi'), ('MYS', 'Malaysia'),
        ('MDV', 'Maldives'), ('MLI', 'Mali'), ('MLT', 'Malta'), ('MHL', 'Marshall Islands'),
        ('MTQ', 'Martinique'), ('MRT', 'Mauritania'), ('MUS', 'Mauritius'), ('MYT', 'Mayotte'),
        ('MEX', 'Mexico'), ('FSM', 'Micronesia, Federated States of'), ('MDA', 'Moldova, Republic of'),
        ('MCO', 'Monaco'), ('MNG', 'Mongolia'), ('MNE', 'Montenegro'), ('MSR', 'Montserrat'),
        ('MAR', 'Morocco'), ('MOZ', 'Mozambique'), ('MMR', 'Myanmar'), ('NAM', 'Namibia'),
        ('NRU', 'Nauru'), ('NPL', 'Nepal'), ('NLD', 'Netherlands'), ('NCL', 'New Caledonia'),
        ('NZL', 'New Zealand'), ('NIC', 'Nicaragua'), ('NER', 'Niger'), ('NGA', 'Nigeria'),
        ('NIU', 'Niue'), ('NFK', 'Norfolk Island'), ('MKD', 'North Macedonia'), ('MNP', 'Northern Mariana Islands'),
        ('NOR', 'Norway'), ('OMN', 'Oman'), ('PAK', 'Pakistan'), ('PLW', 'Palau'),
        ('PSE', 'Palestine, State of'), ('PAN', 'Panama'), ('PNG', 'Papua New Guinea'),
        ('PRY', 'Paraguay'), ('PER', 'Peru'), ('PHL', 'Philippines'), ('PCN', 'Pitcairn'),
        ('POL', 'Poland'), ('PRT', 'Portugal'), ('PRI', 'Puerto Rico'), ('QAT', 'Qatar'),
        ('REU', 'Réunion'), ('ROU', 'Romania'), ('RUS', 'Russian Federation'), ('RWA', 'Rwanda'),
        ('BLM', 'Saint Barthélemy'), ('SHN', 'Saint Helena, Ascension and Tristan da Cunha'),
        ('KNA', 'Saint Kitts and Nevis'), ('LCA', 'Saint Lucia'), ('MAF', 'Saint Martin (French part)'),
        ('SPM', 'Saint Pierre and Miquelon'), ('VCT', 'Saint Vincent and the Grenadines'),
        ('WSM', 'Samoa'), ('SMR', 'San Marino'), ('STP', 'Sao Tome and Principe'),
        ('SAU', 'Saudi Arabia'), ('SEN', 'Senegal'), ('SRB', 'Serbia'), ('SYC', 'Seychelles'),
        ('SLE', 'Sierra Leone'), ('SGP', 'Singapore'), ('SXM', 'Sint Maarten (Dutch part)'),
        ('SVK', 'Slovakia'), ('SVN', 'Slovenia'), ('SLB', 'Solomon Islands'), ('SOM', 'Somalia'),
        ('ZAF', 'South Africa'), ('SGS', 'South Georgia and the South Sandwich Islands'),
        ('SSD', 'South Sudan'), ('ESP', 'Spain'), ('LKA', 'Sri Lanka'), ('SDN', 'Sudan'),
        ('SUR', 'Suriname'), ('SJM', 'Svalbard and Jan Mayen'), ('SWE', 'Sweden'), ('CHE', 'Switzerland'),
        ('SYR', 'Syrian Arab Republic'), ('TWN', 'Taiwan, Province of China'), ('TJK', 'Tajikistan'),
        ('TZA', 'Tanzania, United Republic of'), ('THA', 'Thailand'), ('TLS', 'Timor-Leste'),
        ('TGO', 'Togo'), ('TKL', 'Tokelau'), ('TON', 'Tonga'), ('TTO', 'Trinidad and Tobago'),
        ('TUN', 'Tunisia'), ('TUR', 'Turkey'), ('TKM', 'Turkmenistan'), ('TCA', 'Turks and Caicos Islands'),
        ('TUV', 'Tuvalu'), ('UGA', 'Uganda'), ('UKR', 'Ukraine'), ('ARE', 'United Arab Emirates'),
        ('GBR', 'United Kingdom'), ('USA', 'United States'), ('UMI', 'United States Minor Outlying Islands'),
        ('URY', 'Uruguay'), ('UZB', 'Uzbekistan'), ('VUT', 'Vanuatu'), ('VEN', 'Venezuela'),
        ('VNM', 'Viet Nam'), ('VGB', 'Virgin Islands, British'), ('VIR', 'Virgin Islands, U.S.'),
        ('WLF', 'Wallis and Futuna'), ('ESH', 'Western Sahara'), ('YEM', 'Yemen'), ('ZMB', 'Zambia'),
        ('ZWE', 'Zimbabwe')
    ]
    return countries

# ============================================================================
# GENERATE QUERIES
# ============================================================================

def generate_queries():
    """Generate all audit queries."""
    print("="*70)
    print("GENERATING AUDIT QUERIES")
    print("="*70)
    
    # Get countries
    countries = get_countries_from_dataset()
    print(f"\nCountries: {len(countries)}")
    print(f"Metrics: {len(METRICS)}")
    print(f"Total queries: {len(countries) * len(METRICS):,}")
    
    # Generate queries
    audit_queries = []
    for iso3, country in countries:
        for metric in METRICS:
            # Generate query text
            query = f"What is the {metric} for {country} in 2025?"
            
            audit_queries.append({
                'ISO3': iso3,
                'Country': country,
                'Metric': metric,
                'Query': query,
                'Ground_Truth': None  # Will be filled from dataset if available
            })
    
    print(f"\n✓ Generated {len(audit_queries):,} queries")
    return audit_queries

# ============================================================================
# SAVE TO FILE
# ============================================================================

def save_to_file(audit_queries):
    """Save audit_queries to audit_data.py file."""
    print(f"\nSaving to {OUTPUT_FILE}...")
    
    # Format as Python code
    lines = [
        "# ============================================================================",
        "# AUDIT DATA - Generated Programmatically",
        "# ============================================================================",
        "# This file contains 1,816 audit queries (227 countries × 8 metrics)",
        "# Generated automatically - do not edit manually",
        "# ============================================================================",
        "",
        "audit_queries = ["
    ]
    
    # Add each query
    for query in audit_queries:
        lines.append("    {")
        lines.append(f"        'ISO3': {repr(query['ISO3'])}, ")
        lines.append(f"        'Country': {repr(query['Country'])}, ")
        lines.append(f"        'Metric': {repr(query['Metric'])}, ")
        lines.append(f"        'Query': {repr(query['Query'])}, ")
        lines.append(f"        'Ground_Truth': {repr(query['Ground_Truth'])}, ")
        lines.append("    },")
    
    lines.append("]")
    
    # Write file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"✓ Saved {len(audit_queries):,} queries to {OUTPUT_FILE}")
    print(f"✓ File size: {OUTPUT_FILE.stat().st_size / 1024:.2f} KB")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    audit_queries = generate_queries()
    save_to_file(audit_queries)
    print("\n" + "="*70)
    print("✅ GENERATION COMPLETE")
    print("="*70)
    print(f"✓ {len(audit_queries):,} queries generated")
    print(f"✓ Saved to: {OUTPUT_FILE}")
    print("="*70)
