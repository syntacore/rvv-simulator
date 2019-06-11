.PHONY: all docbook clean

adoc_src:= $(wildcard  ./*.adoc)
pdfs:=$(patsubst %.adoc,%.pdf, $(adoc_src))
xmls:=$(patsubst %.adoc,%.xml, $(adoc_src))

$(pdfs):%.pdf:%.adoc
	asciidoctor-pdf -o $@ $<

all: $(pdfs) docbook

docbook: $(xmls)

$(xmls):%.xml:%.adoc
	asciidoctor -v -b docbook -o $@ $<

clean:
	$(RM) $(pdfs) $(xmls)
