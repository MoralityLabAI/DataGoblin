// Read Inkling's pass-boundary states through the public ggml eval callback.
// Compile against the headers belonging to the installed llama.cpp build.
#include "llama.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

constexpr int WIDTH = 2304;
constexpr int CONTEXT = 1024;
constexpr int BATCH = 16;
constexpr int MAX_PROMPTS = 64;

struct Trace {
    std::ofstream data;
    std::ofstream index;
    std::map<std::string, int> anchors;
    std::string key;
    std::string error;
    int offset = 0;
    int batch = 0;
    int64_t rows = 0;
    bool enabled = false;
    std::map<std::string, bool> seen;
    std::vector<float> readout_delta;
    int intervention_token = -1;
    double realized_dose = 0.0;
    bool intervention_applied = false;

    Trace(const fs::path &out, int passes)
        : data(out / "activations.f32", std::ios::binary), index(out / "rows.tsv") {
        if (!data || !index) {
            throw std::runtime_error("Could not open trace files");
        }
        anchors.emplace("l_out-5", 0);
        anchors.emplace("l_out-17", 1);
        for (int pass = 2; pass <= passes; ++pass) {
            anchors.emplace("l_out-" + std::to_string(35 + (pass - 2) * 12), pass);
        }
        index << "sample_key\tpass_index\ttensor\ttoken_index\n";
    }
};

static bool observe(ggml_tensor *tensor, bool ask, void *data) {
    auto &trace = *static_cast<Trace *>(data);
    if (!trace.enabled) {
        return !ask;
    }
    try {
        const std::string name = ggml_get_name(tensor);
        const auto anchor = trace.anchors.find(name);
        const bool wanted = anchor != trace.anchors.end();
        if (ask || !wanted) {
            return ask ? wanted : true;
        }
        if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor)
                || tensor->ne[0] != WIDTH || ggml_nrows(tensor) != trace.batch
                || !trace.seen.emplace(name, true).second) {
            throw std::runtime_error("Unexpected or duplicate boundary tensor: " + name);
        }
        std::vector<float> block(WIDTH * trace.batch);
        const size_t size = block.size() * sizeof(float);
        ggml_backend_tensor_get(tensor, block.data(), 0, size);
        const int cue = trace.intervention_token - trace.offset;
        if (!trace.readout_delta.empty() && anchor->second == static_cast<int>(trace.anchors.size()) - 1
                && cue >= 0 && cue < trace.batch) {
            if (trace.intervention_applied) {
                throw std::runtime_error("Repeated readout intervention");
            }
            float *row = block.data() + cue * WIDTH;
            double norm_squared = 0.0;
            for (int i = 0; i < WIDTH; ++i) {
                const float original = row[i];
                if (trace.readout_delta[i] != 0) {
                    row[i] += trace.readout_delta[i];
                }
                const double delta = double(row[i]) - original;
                norm_squared += delta * delta;
            }
            ggml_backend_tensor_set(tensor, row, cue * WIDTH * sizeof(float), WIDTH * sizeof(float));
            trace.realized_dose = std::sqrt(norm_squared);
            trace.intervention_applied = true;
        }
        trace.data.write(reinterpret_cast<const char *>(block.data()), size);
        for (int token = 0; token < trace.batch; ++token) {
            trace.index << trace.key << '\t' << anchor->second << '\t' << name << '\t'
                        << trace.offset + token << '\n';
        }
        if (!trace.data || !trace.index) {
            throw std::runtime_error("Writing the activation trace failed");
        }
        trace.rows += trace.batch;
        return true;
    } catch (const std::exception &error) {
        trace.error = error.what();
        return false;
    }
}

static std::vector<std::string> load_cpu_patterns(const fs::path &path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("Could not open the placement plan");
    }
    std::vector<std::string> result;
    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty()) {
            result.push_back(line);
        }
    }
    if (result.size() >= llama_max_tensor_buft_overrides()) {
        throw std::runtime_error("Placement exceeds the runtime's override capacity");
    }
    return result;
}

static std::vector<llama_token> tokenize(const llama_vocab *vocab, const fs::path &path) {
    const auto size = fs::file_size(path);
    if (!size || size > 32768) {
        throw std::runtime_error("Prompt exceeds the registered byte limit");
    }
    std::ifstream prompt(path, std::ios::binary);
    std::string bytes(size, 0);
    if (!prompt.read(bytes.data(), bytes.size())) {
        throw std::runtime_error("Could not read the prompt");
    }
    std::vector<llama_token> tokens(CONTEXT);
    const int count = llama_tokenize(vocab, bytes.data(), static_cast<int>(bytes.size()),
                                     tokens.data(), CONTEXT, true, true);
    if (count <= 0 || count > CONTEXT) {
        throw std::runtime_error("Prompt exceeds the trace context: " + path.filename().string());
    }
    tokens.resize(count);
    return tokens;
}

static std::vector<float> load_delta(const fs::path &path) {
    if (!fs::exists(path)) {
        return {};
    }
    if (fs::file_size(path) != WIDTH * sizeof(float)) {
        throw std::runtime_error("Intervention vector has the wrong width");
    }
    std::vector<float> data(WIDTH);
    std::ifstream stream(path, std::ios::binary);
    if (!stream.read(reinterpret_cast<char *>(data.data()), WIDTH * sizeof(float))
            || std::any_of(data.begin(), data.end(), [](float value) { return !std::isfinite(value); })) {
        throw std::runtime_error("Intervention vector is unreadable or nonfinite");
    }
    return data;
}

static void decode(llama_context *ctx, Trace &trace, const std::vector<llama_token> &tokens) {
    llama_memory_clear(llama_get_memory(ctx), true);
    const int count = static_cast<int>(tokens.size());
    auto batch = llama_batch_init(BATCH, 0, 1);
    int status = 0;
    for (int offset = 0; offset < count; offset += BATCH) {
        trace.offset = offset;
        trace.batch = std::min(BATCH, count - offset);
        trace.seen.clear();
        batch.n_tokens = trace.batch;
        for (int i = 0; i < batch.n_tokens; ++i) {
            batch.token[i] = tokens[offset + i];
            batch.pos[i] = offset + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = i == batch.n_tokens - 1;
        }
        trace.enabled = true;
        status = llama_decode(ctx, batch);
        trace.enabled = false;
        if (status || !trace.error.empty() || trace.seen.size() != trace.anchors.size()) {
            break;
        }
    }
    llama_batch_free(batch);
    if (status || !trace.error.empty() || trace.seen.size() != trace.anchors.size()) {
        throw std::runtime_error("Incomplete decode for " + trace.key + ": " + trace.error);
    }
}

struct Completion {
    std::string raw;
    int tokens = 0;
    bool stopped = false;
};

static Completion generate(llama_context *ctx, const llama_vocab *vocab, int position, int budget) {
    Completion result;
    auto batch = llama_batch_init(1, 0, 1);
    try {
        const int32_t vocab_size = llama_vocab_n_tokens(vocab);
        while (result.tokens < budget) {
            const float *logits = llama_get_logits_ith(ctx, -1);
            if (!logits || vocab_size <= 0) {
                throw std::runtime_error("No logits for native generation");
            }
            const float *best = std::max_element(logits, logits + vocab_size);
            if (!std::isfinite(*best)) {
                throw std::runtime_error("Nonfinite native generation logits");
            }
            const auto token = static_cast<llama_token>(best - logits);
            ++result.tokens;
            if (llama_vocab_is_eog(vocab, token)) {
                result.stopped = true;
                break;
            }
            std::array<char, 256> piece{};
            const int size = llama_token_to_piece(vocab, token, piece.data(),
                                                static_cast<int>(piece.size()), 0, true);
            if (size < 0 || size > static_cast<int>(piece.size())) {
                throw std::runtime_error("A decoded token exceeds the native buffer");
            }
            const std::string text(piece.data(), size);
            if (text == "<|end_message|>") {
                result.stopped = true;
                break;
            }
            result.raw += text;
            if (result.tokens == budget) {
                break;
            }
            batch.n_tokens = 1;
            batch.token[0] = token;
            batch.pos[0] = position++;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = true;
            if (llama_decode(ctx, batch)) {
                throw std::runtime_error("Native generated-token decode failed");
            }
        }
    } catch (...) {
        llama_batch_free(batch);
        throw;
    }
    llama_batch_free(batch);
    return result;
}

int main(int argc, char **argv) {
    if (argc == 2 && std::string(argv[1]) == "--check-binding") {
        const auto model = llama_model_default_params();
        const auto context = llama_context_default_params();
        std::cout << "Load mode " << model.load_mode << ", context " << context.n_ctx << '\n';
        return 0;
    }
    if (argc < 10 || argc > 12) {
        std::cerr << "Use: inkling_trace RUNTIME GGUF DEVICE PASSES GPU_LAYERS CPU_PATTERNS PROMPTS OUT "
                  << "mmap|none [MAX_NEW_TOKENS [INTERVENTION_DIR]]\n";
        return 2;
    }
    try {
        const int passes = std::stoi(argv[4]);
        const int layers = std::stoi(argv[5]);
        const int new_tokens = argc >= 11 ? std::stoi(argv[10]) : 0;
        const fs::path intervention_dir = argc == 12 ? fs::path(argv[11]) : fs::path();
        const fs::path out = argv[8];
        const std::string load_mode = argv[9];
        if (passes < 1 || passes > 6 || layers < 0 || layers > 256 || new_tokens < 0 || new_tokens > 384
                || (load_mode != "mmap" && load_mode != "none") || fs::exists(out)
                || (!intervention_dir.empty() && (!fs::is_directory(intervention_dir) || !new_tokens))) {
            throw std::runtime_error("Invalid loop, placement, load mode, or output path");
        }
        std::vector<fs::path> prompts;
        for (const auto &entry : fs::directory_iterator(argv[7])) {
            if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                prompts.push_back(entry.path());
            }
        }
        if (prompts.empty() || prompts.size() > MAX_PROMPTS) {
            throw std::runtime_error("Expected 1-64 staged prompt files");
        }
        std::sort(prompts.begin(), prompts.end());
        ggml_backend_load_all_from_path(argv[1]);
        llama_backend_init();
        ggml_backend_dev_t devices[2] = {ggml_backend_dev_by_name(argv[3]), nullptr};
        if (!devices[0]) {
            throw std::runtime_error("Requested backend device is unavailable");
        }
        auto patterns = load_cpu_patterns(argv[6]);
        std::vector<llama_model_tensor_buft_override> buffers;
        for (const auto &pattern : patterns) {
            buffers.push_back({pattern.c_str(), ggml_backend_cpu_buffer_type()});
        }
        buffers.push_back({});
        llama_model_kv_override overrides[2] = {};
        overrides[0].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        std::strcpy(overrides[0].key, "inkling2_loop.loop.count");
        overrides[0].val_i64 = passes;
        auto model_params = llama_model_default_params();
        model_params.devices = devices;
        model_params.n_gpu_layers = layers;
        model_params.tensor_buft_overrides = buffers.data();
        model_params.kv_overrides = overrides;
        model_params.use_extra_bufts = false;
        model_params.no_host = true;
        model_params.load_mode = load_mode == "mmap" ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[2], model_params), llama_model_free);
        char architecture[80] = {};
        if (!model || llama_model_n_embd(model.get()) != WIDTH
                || llama_model_meta_val_str(model.get(), "general.architecture", architecture, sizeof(architecture)) <= 0
                || std::string(architecture) != "inkling2_loop") {
            throw std::runtime_error("Expected the 2304-wide Inkling loop architecture");
        }
        auto config = llama_context_default_params();
        config.n_ctx = CONTEXT;
        config.n_batch = BATCH;
        config.n_ubatch = BATCH;
        config.n_threads = 3;
        config.n_threads_batch = 3;
        config.n_seq_max = 1;
        fs::create_directories(out);
        Trace trace(out, passes);
        config.cb_eval = observe;
        config.cb_eval_user_data = &trace;
        std::unique_ptr<llama_context, decltype(&llama_free)> context(
            llama_init_from_model(model.get(), config), llama_free);
        if (!context) {
            throw std::runtime_error("Could not allocate the trace context");
        }
        std::ofstream progress(out / "completed.tsv");
        progress << "sample_key\tprompt_tokens\tgenerated_tokens\tfinish_reason\telapsed_ms\n" << std::flush;
        std::ofstream config_file(out / "native-config.tsv");
        config_file << "passes\tmax_new_tokens\tcontext\n" << passes << '\t' << new_tokens
                    << '\t' << CONTEXT << '\n';
        config_file.close();
        if (!progress || !config_file) {
            throw std::runtime_error("Could not write native execution metadata");
        }
        std::ofstream interventions;
        if (!intervention_dir.empty()) {
            interventions.open(out / "interventions.tsv");
            interventions << "sample_key\tapplied\tprefill_token\trealized_l2\n" << std::flush;
            interventions.precision(17);
            if (!interventions) {
                throw std::runtime_error("Could not open the intervention receipt");
            }
        }
        const auto *vocab = llama_model_get_vocab(model.get());
        for (const auto &prompt : prompts) {
            const auto started = std::chrono::steady_clock::now();
            trace.key = prompt.stem().string();
            const auto tokens = tokenize(vocab, prompt);
            if (tokens.size() + new_tokens > CONTEXT) {
                throw std::runtime_error("Combined prompt and generation budget exceeds context: " + trace.key);
            }
            if (!intervention_dir.empty()) {
                trace.readout_delta = load_delta(intervention_dir / (trace.key + ".delta.f32"));
                trace.intervention_token = static_cast<int>(tokens.size()) - 1;
                trace.realized_dose = 0;
                trace.intervention_applied = false;
            }
            decode(context.get(), trace, tokens);
            if (trace.intervention_applied != !trace.readout_delta.empty()) {
                throw std::runtime_error("The requested readout intervention did not execute");
            }
            Completion answer;
            const char *reason = "prefill";
            if (new_tokens) {
                answer = generate(context.get(), vocab, static_cast<int>(tokens.size()), new_tokens);
                reason = answer.stopped ? "stop" : "length";
                std::ofstream reply(out / (trace.key + ".reply"), std::ios::binary);
                reply.write(answer.raw.data(), answer.raw.size());
                reply.close();
                if (!reply) {
                    throw std::runtime_error("Could not finish writing the native reply");
                }
            }
            trace.data.flush();
            trace.index.flush();
            if (interventions.is_open()) {
                interventions << trace.key << '\t' << int(trace.intervention_applied) << '\t'
                              << trace.intervention_token << '\t' << trace.realized_dose << '\n' << std::flush;
                if (!interventions) {
                    throw std::runtime_error("Could not publish the intervention receipt");
                }
            }
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - started).count();
            progress << trace.key << '\t' << tokens.size() << '\t' << answer.tokens << '\t'
                     << reason << '\t' << elapsed << '\n' << std::flush;
            if (!trace.data || !trace.index || !progress) {
                throw std::runtime_error("Could not commit a native prompt's output");
            }
            std::cout << trace.key << " " << tokens.size() << " tokens\n" << std::flush;
        }
        trace.data.close();
        trace.index.close();
        if (!trace.data || !trace.index || !trace.rows) {
            throw std::runtime_error("Trace output is incomplete");
        }
        std::ofstream receipt(out / "native-complete.txt");
        receipt << trace.rows << '\n';
        return receipt ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
