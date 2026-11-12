fn main() -> std::io::Result<()> {
    // 1. Your existing MEXC Protobuf generation
    let mut config_mexc = prost_build::Config::new();
    config_mexc.default_package_filename("mexc_proto");
    config_mexc.compile_protos(
        &["../mexc-proto/PushDataV3ApiWrapper.proto"],
        &["../mexc-proto"],
    )?;
    println!("cargo:rerun-if-changed=../mexc-proto/PushDataV3ApiWrapper.proto");

    // 2. Your NEW, completely unrelated Protobuf generation
    let mut config_ev = prost_build::Config::new();
    config_ev.default_package_filename("events_proto");
    config_ev.type_attribute(".", "#[derive(serde::Serialize)]");
    config_ev.compile_protos(&["proto/events.proto"], &["proto"])?;
    println!("cargo:rerun-if-changed=proto/events.proto");

    Ok(())
}
