"""
Module d'étude des dangers (Danger Study Module)
Basé sur l'analyse du PDF "3-Etude-dangers-avec-annexes_v2.pdf"

Ce module implémente une méthodologie d'analyse des risques similaire à celle
présentée dans l'étude des dangers pour une installation classée.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json
from datetime import datetime

class DangerStudy:
    """
    Classe principale pour l'étude des dangers
    """

    def __init__(self, installation_name: str, location: str):
        self.installation_name = installation_name
        self.location = location
        self.date = datetime.now()
        self.environment = {}
        self.hazards = []
        self.scenarios = []
        self.risk_assessment = {}

    def characterize_environment(self, data: Dict):
        """
        Caractérisation de l'environnement
        """
        self.environment = {
            'localisation': data.get('localisation', ''),
            'aléas_naturels': data.get('aléas_naturels', {}),
            'aléas_technologiques': data.get('aléas_technologiques', {}),
            'population': data.get('population', {}),
            'environnement': data.get('environnement', {})
        }

    def identify_hazards(self, hazards_list: List[Dict]):
        """
        Identification des sources potentielles d'agression
        """
        self.hazards = hazards_list

    def define_scenarios(self, scenarios: List[Dict]):
        """
        Définition des scénarios d'accidents
        """
        self.scenarios = scenarios

    def assess_risks(self):
        """
        Évaluation des risques selon la méthodologie de l'étude
        """
        # Analyse préliminaire des risques
        erc = []  # Événements redoutés centraux

        for scenario in self.scenarios:
            # Calcul de la probabilité et gravité
            probability = self._calculate_probability(scenario)
            severity = self._calculate_severity(scenario)

            risk_level = self._evaluate_risk(probability, severity)

            self.risk_assessment[scenario['name']] = {
                'probability': probability,
                'severity': severity,
                'risk_level': risk_level,
                'mitigations': scenario.get('mitigations', [])
            }

    def _calculate_probability(self, scenario: Dict) -> float:
        """
        Calcul de la probabilité d'occurrence
        """
        # Implémentation simplifiée basée sur les facteurs du scénario
        base_prob = scenario.get('base_probability', 0.001)
        factors = scenario.get('risk_factors', [])

        # Ajustement selon les facteurs
        adjustment = 1.0
        for factor in factors:
            if factor['type'] == 'multiplicative':
                adjustment *= factor['value']
            elif factor['type'] == 'additive':
                adjustment += factor['value']

        return min(base_prob * adjustment, 1.0)

    def _calculate_severity(self, scenario: Dict) -> str:
        """
        Évaluation de la gravité des conséquences
        """
        consequences = scenario.get('consequences', {})

        # Critères de gravité
        if consequences.get('fatalities', 0) > 0:
            return 'Critique'
        elif consequences.get('injuries', 0) > 10:
            return 'Majeure'
        elif consequences.get('environmental_impact', '') == 'severe':
            return 'Majeure'
        else:
            return 'Mineure'

    def _evaluate_risk(self, probability: float, severity: str) -> str:
        """
        Évaluation du niveau de risque
        """
        if severity == 'Critique':
            if probability > 0.01:
                return 'Très élevé'
            elif probability > 0.001:
                return 'Élevé'
            else:
                return 'Moyen'
        elif severity == 'Majeure':
            if probability > 0.1:
                return 'Élevé'
            else:
                return 'Moyen'
        else:
            return 'Faible'

    def generate_summary(self) -> str:
        """
        Génération du résumé non technique
        """
        summary = f"""
        ÉTUDE DES DANGERS - {self.installation_name}
        Localisation: {self.location}
        Date: {self.date.strftime('%d/%m/%Y')}

        RÉSUMÉ NON TECHNIQUE

        Cette étude des dangers analyse les risques potentiels associés à l'installation.
        L'analyse s'appuie sur une méthodologie structurée comprenant :

        1. Caractérisation de l'environnement
        2. Identification des sources de danger
        3. Définition des scénarios d'accidents
        4. Évaluation des risques

        ENVIRONNEMENT:
        {json.dumps(self.environment, indent=2, ensure_ascii=False)}

        SOURCES DE DANGER IDENTIFIÉES:
        {len(self.hazards)} sources potentielles d'agression identifiées.

        SCÉNARIOS D'ACCIDENTS:
        {len(self.scenarios)} scénarios analysés.

        ÉVALUATION DES RISQUES:
        {json.dumps(self.risk_assessment, indent=2, ensure_ascii=False)}

        CONCLUSION:
        L'étude permet d'identifier les mesures de prévention et de protection nécessaires
        pour assurer la sécurité de l'installation et de son environnement.
        """

        return summary

    def export_report(self, filename: str):
        """
        Export du rapport complet
        """
        report = {
            'installation': self.installation_name,
            'location': self.location,
            'date': self.date.isoformat(),
            'environment': self.environment,
            'hazards': self.hazards,
            'scenarios': self.scenarios,
            'risk_assessment': self.risk_assessment,
            'summary': self.generate_summary()
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

# Fonction de test
def test_danger_study():
    """
    Fonction de test du module d'étude des dangers
    """
    study = DangerStudy("Installation Test", "Contes, France")

    # Caractérisation environnementale
    env_data = {
        'localisation': 'Zone industrielle',
        'aléas_naturels': {
            'sismicité': 'Zone 4 (moyenne)',
            'inondation': 'Faible risque'
        },
        'population': {
            'habitants_proches': 1000,
            'distance_plus_proche': 500
        }
    }
    study.characterize_environment(env_data)

    # Sources de danger
    hazards = [
        {
            'type': 'Naturel',
            'name': 'Séisme',
            'description': 'Risque sismique zone 4'
        },
        {
            'type': 'Technologique',
            'name': 'Incendie',
            'description': 'Risque d\'incendie dans les stockages'
        }
    ]
    study.identify_hazards(hazards)

    # Scénarios
    scenarios = [
        {
            'name': 'Incendie de stockage',
            'description': 'Incendie généralisé dans une cellule de stockage',
            'base_probability': 0.001,
            'risk_factors': [
                {'type': 'multiplicative', 'value': 2.0, 'reason': 'Matériaux inflammables'}
            ],
            'consequences': {
                'fatalities': 0,
                'injuries': 5,
                'environmental_impact': 'moderate'
            },
            'mitigations': ['Système d\'extinction automatique', 'Surveillance']
        }
    ]
    study.define_scenarios(scenarios)

    # Évaluation
    study.assess_risks()

    # Génération du résumé
    summary = study.generate_summary()
    print(summary)

    # Export
    study.export_report('danger_study_test.json')
    print("Rapport exporté vers danger_study_test.json")

if __name__ == "__main__":
    test_danger_study()