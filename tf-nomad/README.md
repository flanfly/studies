```bash
sh-keygen -R $(terraform output -raw master_public_ip)
ssh -f -N -L 4646:10.0.0.2:4646 root@$(terraform output -raw master_public_ip)
```
