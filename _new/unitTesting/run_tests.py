"""
Standalone-Skript zum Ausführen aller Tests vor dem Training.
"""
import sys
import pytest
from pathlib import Path

def run_all_tests():
    """Führt alle Unit-Tests aus und stoppt bei Fehlern."""
    print("=" * 60)
    print("STARTE UNIT-TESTS...")
    print("=" * 60)

    # Bestimme Test-Verzeichnis relativ zu diesem Skript
    test_dir = Path(__file__).parent

    # Führe Tests aus
    test_result = pytest.main([
        str(test_dir),
        "-v",                          # Verbose
        "--tb=short",                  # Kurze Tracebacks
        "--maxfail=3",                 # Stoppe nach 3 Fehlern
    ])

    if test_result != 0:
        print("\n" + "=" * 60)
        print("❌ TESTS FEHLGESCHLAGEN!")
        print("=" * 60)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ ALLE TESTS ERFOLGREICH!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_all_tests()