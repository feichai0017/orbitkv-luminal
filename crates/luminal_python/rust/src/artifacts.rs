use std::collections::HashMap;

/// Artifact configuration: maps artifact names to their key-value parameters.
/// Passed from Python (which queries torch._logging) to Rust at compilation time.
pub type ArtifactConfig = HashMap<String, HashMap<String, String>>;

/// Process all enabled artifacts based on the config dict from Python.
///
/// To add a new artifact:
/// 1. Register it with `torch._logging.register_artifact()` in Python `__init__.py`
/// 2. Add a check in `_build_artifact_config()` in Python `__init__.py`
/// 3. Add a match arm here with a handler function
pub fn process_artifacts(config: &ArtifactConfig) {
    for (name, params) in config {
        match name.as_str() {
            "luminal_hello_world" => handle_hello_world(params),
            _ => {
                log::warn!("Unknown luminal artifact: {}", name);
            }
        }
    }
}

fn handle_hello_world(params: &HashMap<String, String>) {
    let enabled = params.get("enabled").map(|v| v == "true").unwrap_or(false);
    if !enabled {
        return;
    }

    let default_path = std::env::temp_dir().join("luminal_hello.txt");
    let path = params
        .get("output_path")
        .map(std::path::PathBuf::from)
        .unwrap_or(default_path);

    match std::fs::write(&path, "Hello from luminal!\n") {
        Ok(()) => log::info!("luminal_hello_world artifact written to {}", path.display()),
        Err(e) => log::error!(
            "Failed to write hello_world artifact to {}: {}",
            path.display(),
            e
        ),
    }
}
