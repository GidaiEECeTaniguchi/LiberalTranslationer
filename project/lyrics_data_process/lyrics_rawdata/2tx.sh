for f in *.docx; do
  docx2txt "$f" "${f%.docx}.txt"
done
