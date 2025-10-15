#!/usr/bin/env python3
"""Détecte les cycles d'import critiques - Output JSON pour parsing"""

import json
import signal
import sys

from tools.analysis.deps_graph import DependencyAnalyzer


def timeout_handler(signum, frame):
    """Gestion du timeout"""
    output = {
        'status': 'timeout',
        'critical_cycles': -1,
        'non_critical_cycles': -1,
        'message': 'Analysis timeout after 30 seconds',
    }
    print(json.dumps(output))
    sys.exit(2)


def main():
    # Set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # 30 secondes max

    try:
        analyzer = DependencyAnalyzer()
        analyzer.build_graph()

        cycles = analyzer.find_cycles()
        critical = analyzer.critical_cycles()

        output = {
            'status': 'success',
            'critical_cycles': len(critical),
            'non_critical_cycles': len(cycles) - len(critical),
            'critical_details': critical[:5] if critical else [],  # Top 5 pour ne pas flooder
            'message': None,
        }

        if critical:
            output['status'] = 'critical'
            output['message'] = f"{len(critical)} critical import cycles found"
            print(json.dumps(output))
            sys.exit(1)
        elif cycles:
            output['status'] = 'warning'
            output['message'] = f"{len(cycles)} non-critical cycles found"
            print(json.dumps(output))
            sys.exit(0)
        else:
            output['message'] = "No import cycles detected"
            print(json.dumps(output))
            sys.exit(0)

    except Exception as e:
        output = {'status': 'error', 'critical_cycles': -1, 'non_critical_cycles': -1, 'message': str(e)}
        print(json.dumps(output))
        sys.exit(3)
    finally:
        signal.alarm(0)  # Cancel timeout


if __name__ == "__main__":
    main()
