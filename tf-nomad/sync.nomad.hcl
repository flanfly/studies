job "sync-binance-spot" {
  type = "batch"

  group "sync-datastore" {
    count = 10

    constraint {
      operator = "distinct_hosts"
      value    = "true"
    }

    task "init" {
      driver = "exec"

      lifecycle {
        hook    = "prestart"
        sidecar = false
      }

      artifact {
        source      = "http://10.0.0.2/queue"
        destination = "local/queue"
        mode        = "file"
      }

      config {
        command = "sh"
        args = [
          "-c",
          <<-EOT
            echo "selecting ${NOMAD_ALLOC_INDEX}"
            cat local/queue | sed -n "$((NOMAD_ALLOC_INDEX + 1))p" > alloc/item
          EOT
        ]
      }
    }

    task "sync" {
      driver = "docker"

      env {
        GIT_KEY_FILE = "/secrets/git.key"
      }

      resources {
        memory = 3500
        cpu    = 2000
      }

      artifact {
        source      = "http://10.0.0.2/git.key"
        destination = "secrets/git.key"
        mode        = "file"
      }

      config {
        image = "ghcr.io/flanfly/studies:master@sha256:d268e03f2ec5201d9cd7b41868096669ffc0a02358b383ad9350752c7d7c4824"
        args  = ["sh", "-c", "uv run sync-datastore.py $(cat /alloc/item)"]
      }
    }
  }
}
