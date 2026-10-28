// Ghidra plugin script: send the function under the cursor to the fnname
// service, pick one of the suggestions, and rename the function.
//
// Install: Script Manager -> Script Directories -> add this folder. Bind it to
// a key (e.g. Ctrl-Shift-N). Set FNNAME_URL if the service isn't on localhost.
//
// @category FnName
// @keybinding ctrl shift N
// @menupath Tools.FnName.Suggest Name

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.listing.Function;
import ghidra.program.model.symbol.SourceType;

public class SuggestName extends GhidraScript {

    private static final Pattern CANDIDATE =
            Pattern.compile("\\{\"name\":\"((?:\\\\.|[^\"\\\\])*)\",\"score\":([0-9.eE+-]+)\\}");

    @Override
    public void run() throws Exception {
        Function fn = getFunctionContaining(currentAddress);
        if (fn == null) {
            popup("Put the cursor inside a function first.");
            return;
        }

        DecompInterface decomp = new DecompInterface();
        decomp.openProgram(currentProgram);
        DecompileResults res = decomp.decompileFunction(fn, 60, monitor);
        decomp.dispose();
        if (res == null || !res.decompileCompleted()) {
            popup("Decompilation failed: " + (res == null ? "" : res.getErrorMessage()));
            return;
        }

        String base = System.getenv().getOrDefault("FNNAME_URL", "http://127.0.0.1:8000");
        String body = "{\"code\":" + quote(res.getDecompiledFunction().getC()) + ",\"top\":5}";
        HttpResponse<String> resp = HttpClient.newHttpClient().send(
                HttpRequest.newBuilder(URI.create(base + "/predict"))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(body)).build(),
                HttpResponse.BodyHandlers.ofString());
        if (resp.statusCode() != 200) {
            popup("fnname service returned " + resp.statusCode() + ": " + resp.body());
            return;
        }

        List<String> choices = new ArrayList<>();
        Matcher m = CANDIDATE.matcher(resp.body().replace(" ", ""));
        while (m.find()) {
            choices.add(String.format("%s  (%.0f%%)", m.group(1), 100 * Double.parseDouble(m.group(2))));
        }
        if (choices.isEmpty()) {
            popup("No suggestions for " + fn.getName());
            return;
        }

        String pick = askChoice("FnName", "Rename " + fn.getName() + " to:", choices, choices.get(0));
        String name = pick.substring(0, pick.indexOf("  ("));
        fn.setName(name, SourceType.USER_DEFINED);
        fn.setComment("fnname suggestion: " + pick + "\nall: " + String.join(", ", choices));
        println("Renamed to " + name);
    }

    private static String quote(String s) {
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
