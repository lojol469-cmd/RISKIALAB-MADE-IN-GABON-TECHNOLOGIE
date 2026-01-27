#!/usr/bin/env python3
"""
Script pour nettoyer les méthodes dupliquées dans danger_rag_system.py
"""

def clean_duplicate_methods():
    # Lire le fichier
    with open('danger_rag_system.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Trouver les indices des méthodes dupliquées
    duplicate_methods = []
    method_starts = []

    for i, line in enumerate(lines):
        if line.strip().startswith('def _create_comprehensive_annexes('):
            method_starts.append(i)
        elif line.strip().startswith('def _create_case_studies_section('):
            method_starts.append(i)

    print(f"Méthodes trouvées aux lignes: {method_starts}")

    # Garder seulement la première occurrence de chaque méthode
    # Pour _create_comprehensive_annexes, garder celle à la ligne 1378 (index ~1377)
    # Pour _create_case_studies_section, garder celle à la ligne 1237 (index ~1236)

    # Identifier les blocs à supprimer
    blocks_to_remove = []

    # Trouver le bloc de la deuxième _create_comprehensive_annexes (ligne ~2969)
    start_idx = None
    for i, line in enumerate(lines):
        if i >= 2968 and line.strip().startswith('def _create_comprehensive_annexes('):
            start_idx = i
            break

    if start_idx:
        # Trouver la fin de cette méthode (prochaine méthode ou fin du fichier)
        end_idx = len(lines)
        for i in range(start_idx + 1, len(lines)):
            if lines[i].strip().startswith('def ') and not lines[i].strip().startswith('def _'):
                end_idx = i
                break

        blocks_to_remove.append((start_idx, end_idx))
        print(f"Bloc à supprimer: lignes {start_idx+1} à {end_idx}")

    # Supprimer les blocs en ordre inverse
    for start, end in reversed(blocks_to_remove):
        del lines[start:end]

    # Écrire le fichier nettoyé
    with open('danger_rag_system_clean.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print("Fichier nettoyé créé: danger_rag_system_clean.py")

if __name__ == "__main__":
    clean_duplicate_methods()