job "sync-binance-spot" {
  type = "batch"

  group "sync-datastore" {
    count = 19

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
        image = "ghcr.io/flanfly/studies@sha256:1305fd081e9a8c33391e27c2e0619d1793664e21500990c8bbc9ed7838bb7d19"
        args  = ["sh", "-c", "uv run sync-datastore.py -s $(cat /alloc/item)"]
      }
    }
  }
}
