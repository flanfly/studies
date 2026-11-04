fn main() -> std::io::Result<()> {
    let mut config = prost_build::Config::new();
    config.default_package_filename("mexc_proto");
    config.compile_protos(
        &["../mexc-proto/PushDataV3ApiWrapper.proto"],
        &["../mexc-proto"],
    )
}
