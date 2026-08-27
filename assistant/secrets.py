from __future__ import annotations

import argparse
import getpass
import os

from assistant.secret_storage import TAVILY_SECRET_NAME, WindowsCredentialManagerSecretStorage, sanitize_secret_error


ALLOWED_SECRET_NAMES = {TAVILY_SECRET_NAME}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gerir segredos locais do Echo em armazenamento seguro.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    set_parser = subparsers.add_parser("set", help="Guardar um segredo.")
    set_parser.add_argument("name", choices=sorted(ALLOWED_SECRET_NAMES))

    remove_parser = subparsers.add_parser("remove", help="Remover um segredo.")
    remove_parser.add_argument("name", choices=sorted(ALLOWED_SECRET_NAMES))

    status_parser = subparsers.add_parser("status", help="Verificar se um segredo existe.")
    status_parser.add_argument("name", choices=sorted(ALLOWED_SECRET_NAMES))

    args = parser.parse_args(argv)
    storage = WindowsCredentialManagerSecretStorage()
    if not storage.is_available():
        print("O armazenamento seguro não está disponível neste sistema.")
        return 1

    try:
        if args.command == "set":
            value = getpass.getpass(f"{args.name}: ").strip()
            if not value:
                print("Nenhum valor guardado.")
                return 1
            storage.set_secret(args.name, value)
            print(f"{args.name} guardada em armazenamento seguro.")
            return 0
        if args.command == "remove":
            storage.delete_secret(args.name)
            print(f"{args.name} removida do armazenamento seguro.")
            return 0
        if args.command == "status":
            source = "ambiente" if str(os.environ.get(args.name) or "").strip() else ""
            configured = bool(source) or storage.has_secret(args.name)
            if configured:
                print(f"{args.name} está configurada" + (f" via {source}." if source else " em armazenamento seguro."))
            else:
                print(f"{args.name} não está configurada.")
            return 0
    except Exception as exc:
        print(sanitize_secret_error(exc))
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
