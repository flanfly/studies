locals {
  prefix       = "${random_password.car.result}${random_password.cdr.result}"
  location     = "hel1"
  ssh_key_name = "kai@Kais-MacBook-Pro.local"

  group_name = "${local.prefix}-cluster"

  master_type = "cax11"
  master_ip   = "10.0.0.2"

  worker_type = "cax11"
  workers     = 6
}

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

variable "tailscale_key" {
  description = "Tailscale Auth Key for master"
  type        = string
  sensitive   = true
}

resource "random_password" "car" {
  length  = 1
  special = false
  numeric = false
  upper   = false
}

resource "random_password" "cdr" {
  length  = 4
  special = false
  upper   = false
}

provider "hcloud" {
  token = var.hcloud_token
}

data "hcloud_ssh_key" "user" {
  name = local.ssh_key_name
}

resource "hcloud_ssh_key" "provisioner" {
  name       = "${local.prefix}-key"
  public_key = file("id_ed25519.pub")
}

resource "hcloud_placement_group" "default" {
  name = local.group_name
  type = "spread"
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
  ssh_keys = [
    data.hcloud_ssh_key.user.id,
    hcloud_ssh_key.provisioner.id,
  ]
  placement_group_id = hcloud_placement_group.default.id

  network {
    network_id = hcloud_network.default.id
    ip         = local.master_ip
  }
}

resource "hcloud_server" "workers" {
  count = local.workers

  name        = "${local.prefix}-worker-${count.index + 1}"
  image       = "ubuntu-24.04"
  server_type = local.worker_type
  location    = local.location
  ssh_keys = [
    data.hcloud_ssh_key.user.id,
    hcloud_ssh_key.provisioner.id,
  ]
  placement_group_id = hcloud_placement_group.default.id

  network {
    network_id = hcloud_network.default.id
    ip         = "10.0.0.${count.index + 128}"
  }

  depends_on = [hcloud_network_subnet.default]
}

resource "null_resource" "master_ctor" {
  connection {
    type        = "ssh"
    user        = "root"
    private_key = file("id_ed25519")
    host        = hcloud_server.master.ipv4_address
  }

  triggers = {
    master_id         = hcloud_server.master.id
    playbook_sha256   = filesha256("${path.module}/master.playbook.yml"),
    master_private_ip = local.master_ip
    master_public_ip  = hcloud_server.master.ipv4_address
    tailscale_key     = var.tailscale_key

  }

  provisioner "remote-exec" {
    inline = ["echo 'SSH is ready!'"]
  }

  provisioner "local-exec" {
    command = <<EOT
    ANSIBLE_HOST_KEY_CHECKING=False ansible-playbook \
      -i '${hcloud_server.master.ipv4_address},' \
      --extra-vars 'master_ip=${local.master_ip}' \
      --extra-vars 'hostname=${local.prefix}-master' \
      --extra-vars 'tailscale_key=${var.tailscale_key}' \
      -u root \
      --private-key id_ed25519 \
      master.playbook.yml
    EOT
  }

  depends_on = [hcloud_server.master]
}

resource "null_resource" "master_dtor" {
  triggers = {
    ipv4_address = hcloud_server.master.ipv4_address
  }

  provisioner "local-exec" {
    when    = destroy
    command = "ssh-keygen -f \"$HOME/.ssh/known_hosts\" -R ${self.triggers.ipv4_address}"
  }

  depends_on = [hcloud_server.master]
}

resource "null_resource" "worker_ctor" {
  for_each = {
    for name, s in hcloud_server.workers : name => s
  }

  connection {
    type        = "ssh"
    user        = "root"
    private_key = file("id_ed25519")
    host        = each.value.ipv4_address
  }

  triggers = {
    master_id       = hcloud_server.master.id
    worker_id       = each.value.id
    playbook_sha256 = filesha256("${path.module}/worker.playbook.yml"),
  }

  provisioner "remote-exec" {
    inline = ["echo 'SSH is ready!'"]
  }

  provisioner "local-exec" {
    command = <<EOT
    ANSIBLE_HOST_KEY_CHECKING=False ansible-playbook \
      -i '${each.value.ipv4_address},' \
      --extra-vars 'master_ip=${local.master_ip}' \
      --extra-vars 'hostname=${each.value.name}' \
      -u root \
      --private-key id_ed25519 \
      worker.playbook.yml
    EOT
  }

  depends_on = [hcloud_server.master]
}

output "master_public_ip" {
  value = hcloud_server.master.ipv4_address
}

output "worker_public_ips" {
  value = { for name, s in hcloud_server.workers : name => s.ipv4_address }
}
