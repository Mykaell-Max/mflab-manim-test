from pathlib import Path
import h5py
import numpy as np


FILE = Path(
    r"C:\Users\max00\Downloads\output\output"
    r"\ns_output_ct.000001000.hdf5"
)


def summarize(name, obj):
    indent = "  " * name.count("/")

    if isinstance(obj, h5py.Group):
        print(f"{indent}[GRUPO]   /{name}")

        for key, value in obj.attrs.items():
            print(f"{indent}  atributo {key}: {value}")

    elif isinstance(obj, h5py.Dataset):
        print(
            f"{indent}[DATASET] /{name}\n"
            f"{indent}  shape: {obj.shape}\n"
            f"{indent}  dtype: {obj.dtype}"
        )

        for key, value in obj.attrs.items():
            print(f"{indent}  atributo {key}: {value}")

        # Mostra alguns valores somente para datasets pequenos.
        if obj.size <= 20:
            try:
                print(f"{indent}  valores: {obj[...]}")
            except Exception as error:
                print(f"{indent}  valores indisponíveis: {error}")


print(f"Arquivo: {FILE}")
print(f"Tamanho: {FILE.stat().st_size / 1024**2:.2f} MiB\n")

with h5py.File(FILE, "r") as hdf:
    print("Atributos da raiz:")

    if not hdf.attrs:
        print("  nenhum")

    for key, value in hdf.attrs.items():
        print(f"  {key}: {value}")

    print("\nEstrutura:")
    hdf.visititems(summarize)