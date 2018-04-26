function i = normalize(img)
    maxVal = max(max(img));
    minVal = min(min(img));
    delta = maxVal-minVal;
    i = (img-minVal)./delta;
end