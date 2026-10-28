// Headless post-script: decompile every non-thunk, non-external function and
// write one JSON object per line to the path given as the first script arg.
//
//   analyzeHeadless <proj_dir> tmp -import bin.stripped -deleteProject \
//       -postScript ExportDecompiled.java out.jsonl
//
// @category FnName

import java.io.FileWriter;
import java.io.PrintWriter;

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.database.mem.AddressSourceInfo;
import ghidra.program.model.listing.Function;
import ghidra.program.model.mem.MemoryBlock;

public class ExportDecompiled extends GhidraScript {

    private static final int TIMEOUT_SECONDS = 30;

    @Override
    public void run() throws Exception {
        String outPath = getScriptArgs()[0];
        DecompInterface decomp = new DecompInterface();
        decomp.openProgram(currentProgram);

        int ok = 0, failed = 0;
        try (PrintWriter out = new PrintWriter(new FileWriter(outPath))) {
            for (Function fn : currentProgram.getFunctionManager().getFunctions(true)) {
                if (monitor.isCancelled()) break;
                if (fn.isThunk() || fn.isExternal()) continue;
                MemoryBlock block = getMemoryBlock(fn.getEntryPoint());
                if (block == null || !block.isExecute() || block.getName().startsWith(".plt")) continue;

                AddressSourceInfo src = currentProgram.getMemory().getAddressSourceInfo(fn.getEntryPoint());
                if (src == null || src.getFileOffset() < 0) continue;

                DecompileResults res = decomp.decompileFunction(fn, TIMEOUT_SECONDS, monitor);
                if (res == null || !res.decompileCompleted()) {
                    failed++;
                    continue;
                }
                out.println("{\"ghidra_name\":" + q(fn.getName())
                        + ",\"file_offset\":" + src.getFileOffset()
                        + ",\"n_instructions\":" + countInstructions(fn)
                        + ",\"code\":" + q(res.getDecompiledFunction().getC()) + "}");
                ok++;
            }
        } finally {
            decomp.dispose();
        }
        println("ExportDecompiled: " + ok + " functions, " + failed + " failed");
    }

    private int countInstructions(Function fn) {
        int n = 0;
        var it = currentProgram.getListing().getInstructions(fn.getBody(), true);
        while (it.hasNext()) { it.next(); n++; }
        return n;
    }

    private static String q(String s) {
        StringBuilder b = new StringBuilder("\"");
        for (char c : s.toCharArray()) {
            switch (c) {
                case '"': b.append("\\\""); break;
                case '\\': b.append("\\\\"); break;
                case '\n': b.append("\\n"); break;
                case '\r': b.append("\\r"); break;
                case '\t': b.append("\\t"); break;
                default:
                    if (c < 0x20) b.append(String.format("\\u%04x", (int) c));
                    else b.append(c);
            }
        }
        return b.append('"').toString();
    }
}
