# papers/%.pdf --marker--> papers/.extracted/%/ --goose--> summaries/%.md
# Subdirectories are mirrored: papers/crypto/smith.pdf -> summaries/crypto/smith.md
#
MARKER = uvx --from 'marker-pdf[all]' marker_single "$<" --mode fast --output_format markdown --output_dir "papers/.extracted/$(dir $*)"
GOOSE  = goose run --recipe recipes/extract-markdown.yaml --params "input_path=papers/.extracted/$*/" --params "output_file=$@"

PDFS      := $(shell find papers -name '*.pdf' -not -path '*/.*' 2>/dev/null)
SUMMARIES := $(patsubst papers/%.pdf,summaries/%.md,$(PDFS))

.PHONY: all
all: rename
	@$(MAKE) --no-print-directory summaries

.PHONY: summaries
summaries: $(SUMMARIES)

# % matches slashes in GNU make, so these handle nested paths unchanged.
papers/.extracted/%/stamp: papers/%.pdf
	@rm -rf $(dir $@)
	@mkdir -p papers/.extracted/$(dir $*)
	$(MARKER)
	@test -n "$$(find $(dir $@) -name '*.md' -print -quit 2>/dev/null)" \
		|| { echo "ERROR: no markdown under $(dir $@) — did marker change its output layout?" >&2; exit 1; }
	@touch $@

summaries/%.md: papers/.extracted/%/stamp
	@mkdir -p $(dir $@)
	@rm -f $@
	$(GOOSE)

.PHONY: clean
clean:
	rm -rf papers/.extracted summaries

.PHONY: rename
# Directories first and deepest-first (-depth), so renaming a parent cannot
# invalidate paths still queued for its children.
rename:
	@find papers -depth -mindepth 1 -type d -not -path '*/.*' | while IFS= read -r d; do \
	  b=$$(basename "$$d"); p=$$(dirname "$$d"); \
	  s=$$(printf '%s' "$$b" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' \
	       | sed 's/^-*//; s/-*$$//'); \
	  [ "$$b" = "$$s" ] && continue; \
	  if [ -e "$$p/$$s" ]; then echo "skip, target exists: $$d" >&2; continue; fi; \
	  echo "  $$d/ -> $$p/$$s/"; \
	  mv "$$d" "$$p/$$s"; \
	done
	@find papers -name '*.pdf' -not -path '*/.*' | while IFS= read -r f; do \
	  b=$$(basename "$$f" .pdf); p=$$(dirname "$$f"); \
	  s=$$(printf '%s' "$$b" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' \
	       | sed 's/^-*//; s/-*$$//'); \
	  [ "$$b" = "$$s" ] && continue; \
	  if [ -e "$$p/$$s.pdf" ]; then echo "skip, target exists: $$f" >&2; continue; fi; \
	  echo "  $$f -> $$p/$$s.pdf"; \
	  mv "$$f" "$$p/$$s.pdf"; \
	done

# A killed goose run leaves a truncated summary that make records as done.
.DELETE_ON_ERROR:
# Stamps are intermediates in a chained rule; without this make deletes them
# and re-runs marker on every build.
.SECONDARY:
