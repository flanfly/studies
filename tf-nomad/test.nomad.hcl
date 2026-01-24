job "docs" {
  type = "batch"

  group "example" {
    count = 19

    constraint {
      operator = "distinct_hosts"
      value    = "true"
    }

    task "uptime" {
      driver = "exec"

      template {
        data        = <<-EOT
          alpha
          beta
          gamma
          delta
          epsilon
          zeta
          eta
          theta
          iota
          kappa
          lambda
          mu
          nu
          xi
          omicron
          pi
          rho
          sigma
          tau
        EOT
        destination = "local/queue"
      }

      config {
        command = "sh"
        args = [
          "-c",
          <<-EOT
            echo "Processing task ${NOMAD_ALLOC_INDEX}"
            echo "$(cat local/queue | sed -n "$((NOMAD_ALLOC_INDEX + 1))p")"
            uptime
            echo "Done with task ${NOMAD_ALLOC_INDEX}"
          EOT
        ]
      }
    }
  }
}

