terraform {
  required_providers {
    hcloud = {
      source  = "hetznercloud/hcloud"
      version = "~> 1.45"
    }
  }
}

variable "hcloud_token" {
  description = "Hetzner Cloud API token"
  type        = string
  sensitive   = true
}

resource "random_password" "car" {
  length  = 1
  special = false
  numeric = false
}

resource "random_password" "cdr" {
  length  = 4
  special = false
}

locals {
  prefix       = "${random_password.car.result}${random_password.cdr.result}"
  location     = "hel1"
  ssh_key_name = "kai@Kais-MacBook-Pro.local"

  master_type = "cax11"
  master_ip   = "10.0.0.2"

  worker_type = "cax11"
  workers     = 3
}

provider "hcloud" {
  token = var.hcloud_token
}

data "hcloud_ssh_key" "default" {
  name = local.ssh_key_name
}

resource "hcloud_network" "default" {
  name     = "${local.prefix}-net"
  ip_range = "10.0.0.0/16"
}

resource "hcloud_network_subnet" "default" {
  network_id   = hcloud_network.default.id
  type         = "cloud"
  network_zone = "eu-central"
  ip_range     = "10.0.0.0/24"
}

resource "hcloud_server" "master" {
  name        = "${local.prefix}-master"
  image       = "ubuntu-24.04"
  server_type = local.master_type
  location    = local.location
  ssh_keys    = [data.hcloud_ssh_key.default.id]

  network {
    network_id = hcloud_network.default.id
    ip         = local.master_ip
  }

  user_data = templatefile("${path.module}/master.cloud-init.yaml", {
    master_ip = "10.0.0.2"
    hostname  = "${local.prefix}-master"
  })
}

resource "hcloud_server" "workers" {
  count = local.workers

  name        = "${local.prefix}-worker-${count.index + 1}"
  image       = "ubuntu-24.04"
  server_type = local.worker_type
  location    = local.location
  ssh_keys    = [data.hcloud_ssh_key.default.id]

  network {
    network_id = hcloud_network.default.id
    ip         = "10.0.0.${count.index + 128}"
  }

  user_data = templatefile("${path.module}/worker.cloud-init.yaml", {
    master_ip = "10.0.0.2"
    hostname  = "${local.prefix}-worker-${count.index + 1}"
  })

  depends_on = [hcloud_network_subnet.default]
}

output "master_public_ip" {
  value = hcloud_server.master.ipv4_address
}

output "worker_public_ips" {
  value = { for name, s in hcloud_server.workers : name => s.ipv4_address }
}
