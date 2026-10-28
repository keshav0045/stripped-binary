"""Ground-truth labels from ELF symbol tables.

`strip` removes .symtab but leaves PT_LOAD segments where they were, so a
function's virtual address is the same in the stripped and unstripped copy.
Ghidra reports a file offset for each function in the stripped binary; we turn
that back into a vaddr with the program headers and look the name up here.
"""

from elftools.elf.elffile import ELFFile


def function_symbols(path):
    """{vaddr: (name, size)} for every defined STT_FUNC symbol."""
    out = {}
    with open(path, "rb") as f:
        elf = ELFFile(f)
        for section_name in (".symtab", ".dynsym"):
            sec = elf.get_section_by_name(section_name)
            if sec is None:
                continue
            for sym in sec.iter_symbols():
                if sym["st_info"]["type"] != "STT_FUNC" or sym["st_shndx"] == "SHN_UNDEF":
                    continue
                # .symtab wins over .dynsym: it has local (static) functions too
                out.setdefault(sym["st_value"], (sym.name, sym["st_size"]))
    return out


def offset_to_vaddr(path):
    """Return a function mapping a file offset to a vaddr via PT_LOAD headers."""
    with open(path, "rb") as f:
        segs = [
            (s["p_offset"], s["p_filesz"], s["p_vaddr"])
            for s in ELFFile(f).iter_segments()
            if s["p_type"] == "PT_LOAD"
        ]

    def convert(off):
        for start, size, vaddr in segs:
            if start <= off < start + size:
                return vaddr + off - start
        return None

    return convert
